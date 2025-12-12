"""
Trainer for MU Transformer
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import math

from .losses import MUTransformerLoss, LanguageModelingLoss
from .scheduler import get_scheduler
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import setup_logger, MetricsLogger


class Trainer:
    """
    Trainer for MU Transformer and Baseline models

    Args:
        model: Model to train
        config: Training configuration dictionary
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup loss
        self.criterion = self._setup_loss()

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', 'results/checkpoints'),
            max_checkpoints=config.get('max_checkpoints', 5)
        )

        # Setup logging
        self.logger = setup_logger(
            name='trainer',
            log_file=config.get('log_file', 'results/logs/training.log')
        )

        self.metrics_logger = MetricsLogger(
            log_dir=config.get('log_dir', 'results/logs')
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed precision training
        self.use_amp = config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Gradient clipping
        self.max_grad_norm = config.get('gradient_clip', 1.0)

        # Logging intervals
        self.log_interval = config.get('log_interval', 100)
        self.eval_interval = config.get('eval_interval', 1000)
        self.save_interval = config.get('save_interval', 1000)

    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 3e-4)
        weight_decay = self.config.get('weight_decay', 0.01)

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=self.config.get('betas', (0.9, 0.999)),
                eps=self.config.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=self.config.get('betas', (0.9, 0.999)),
                eps=self.config.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _setup_loss(self):
        """Setup loss function"""
        # Check if model is MU Transformer
        model_name = self.model.__class__.__name__
        is_mu = 'MU' in model_name

        if is_mu:
            # Use combined loss for MU models
            return MUTransformerLoss(
                lambda_lm=self.config.get('lambda_lm', 1.0),
                lambda_inv=self.config.get('lambda_inv', 1.0),
                ignore_index=-100
            )
        else:
            # Use standard LM loss for baseline
            return LanguageModelingLoss(ignore_index=-100)

    def setup_scheduler(self, total_steps: int):
        """
        Setup learning rate scheduler

        Args:
            total_steps: Total number of training steps
        """
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=self.config.get('scheduler', 'cosine_warmup'),
            warmup_steps=self.config.get('warmup_steps', 500),
            total_steps=total_steps,
            min_lr_ratio=self.config.get('min_lr_ratio', 0.0)
        )

    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, MU = self.model(input_ids)

                # Compute loss
                if isinstance(self.criterion, MUTransformerLoss):
                    loss, loss_dict = self.criterion(logits, labels, MU)
                else:
                    loss = self.criterion(logits, labels)
                    loss_dict = {'total': loss.item(), 'lm': loss.item()}

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Regular training
            logits, MU = self.model(input_ids)

            # Compute loss
            if isinstance(self.criterion, MUTransformerLoss):
                loss, loss_dict = self.criterion(logits, labels, MU)
            else:
                loss = self.criterion(logits, labels)
                loss_dict = {'total': loss.item(), 'lm': loss.item()}

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()

        # Scheduler step
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        # Compute perplexity
        perplexity = math.exp(min(loss_dict['lm'], 20))  # Cap for numerical stability

        return {
            **loss_dict,
            'perplexity': perplexity,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on dataloader

        Args:
            dataloader: Evaluation dataloader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_lm_loss = 0.0
        total_inv_loss = 0.0
        total_samples = 0

        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits, MU = self.model(input_ids)

            # Compute loss
            if isinstance(self.criterion, MUTransformerLoss):
                loss, loss_dict = self.criterion(logits, labels, MU)
            else:
                loss = self.criterion(logits, labels)
                loss_dict = {'total': loss.item(), 'lm': loss.item(), 'invariance': 0.0}

            batch_size = input_ids.size(0)
            total_loss += loss_dict['total'] * batch_size
            total_lm_loss += loss_dict['lm'] * batch_size
            total_inv_loss += loss_dict['invariance'] * batch_size
            total_samples += batch_size

        # Average losses
        avg_loss = total_loss / total_samples
        avg_lm_loss = total_lm_loss / total_samples
        avg_inv_loss = total_inv_loss / total_samples

        # Compute perplexity
        perplexity = math.exp(min(avg_lm_loss, 20))

        return {
            'loss': avg_loss,
            'lm_loss': avg_lm_loss,
            'inv_loss': avg_inv_loss,
            'perplexity': perplexity
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Train model

        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            num_epochs: Number of epochs (if None, use config)
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 10)

        # Calculate total steps
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch

        # Setup scheduler
        self.setup_scheduler(total_steps)

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training loop
            epoch_metrics = []
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

            for batch_idx, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['total']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{metrics['lr']:.2e}"
                })

                # Log metrics
                if self.global_step % self.log_interval == 0:
                    self.metrics_logger.log(
                        step=self.global_step,
                        train_loss=metrics['total'],
                        train_perplexity=metrics['perplexity'],
                        learning_rate=metrics['lr']
                    )

                # Evaluate
                if val_loader is not None and self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate(val_loader)
                    self.logger.info(
                        f"Step {self.global_step} - Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Perplexity: {val_metrics['perplexity']:.2f}"
                    )

                    self.metrics_logger.log(
                        step=self.global_step,
                        val_loss=val_metrics['loss'],
                        val_perplexity=val_metrics['perplexity']
                    )

                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=self.global_step,
                            metrics=val_metrics,
                            config=self.config,
                            is_best=True
                        )

                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.global_step,
                        metrics=metrics,
                        config=self.config
                    )

            # Epoch summary
            avg_epoch_loss = sum(m['total'] for m in epoch_metrics) / len(epoch_metrics)
            avg_epoch_ppl = sum(m['perplexity'] for m in epoch_metrics) / len(epoch_metrics)

            self.logger.info(
                f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}, "
                f"Avg Perplexity: {avg_epoch_ppl:.2f}"
            )

            # Evaluate at end of epoch
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.logger.info(
                    f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Perplexity: {val_metrics['perplexity']:.2f}"
                )

        # Save metrics
        self.metrics_logger.save('final_metrics.txt')
        self.logger.info("Training completed!")
