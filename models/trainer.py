import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.optim as optim
import time
import numpy as np
from collections import OrderedDict, defaultdict

from models.utils import *
from utils.eval_funcs import evaluation
from utils import new_utils


class RVQVAETrainer(object):
    def __init__(self, args, vq_model):
        self.args = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.optimizer = optim.AdamW(
                self.vq_model.parameters(), 
                lr=self.args.lr, 
                betas=(0.9, 0.99), 
                weight_decay=self.args.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, 
                milestones=self.args.milestones, 
                gamma=self.args.gamma)
            
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()

            self.logger = new_utils.get_logger(args.out_dir)
            self.writer = SummaryWriter(args.out_dir)
    
    def save(self, file_name, epoch, iter):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'epoch': epoch,
            'iter': iter,
        }
        torch.save(state, file_name)
    
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr    
    
    def forward(self, batch_data):
        motion = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motion)

        loss_rec = self.l1_criterion(pred_motion, motion)
        pred_local_pos = pred_motion[..., 4 : (self.args.nb_joints - 1) * 3 + 4]
        local_pos = motion[..., 4 : (self.args.nb_joints - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        loss = loss_rec + self.args.vel * loss_explicit + self.args.commit * loss_commit

        return loss, loss_rec, loss_explicit, loss_commit, perplexity

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.vq_model.to(self.device)

        start_time = time.time()
        max_iter = self.args.max_epoch * len(train_loader)
        self.logger.info(f'Total Epochs: {self.args.max_epoch}, Total Iters: {max_iter}')
        self.logger.info('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(val_loader)))
        
        epoch = 0
        iter = 0        
        
        logs = defaultdict(def_value, OrderedDict())

        best_metrics = {
            'fid': 100,
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'matching': 100,
            'div': 100,
            'mm': 0
        }

        self.logger.info('First evaluation:')
        best_metrics = evaluation(
            eval_val_loader, eval_wrapper, 'vq_model', self.vq_model,
            logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
            out_dir=self.args.out_dir)

        while epoch < self.args.max_epoch:
            epoch += 1

            self.logger.info(f'Training: {epoch}')
            self.vq_model.train()

            for i, batch_data in enumerate(train_loader):
                iter += 1
                if iter < self.args.warm_up_iter:
                    self.update_lr_warm_up(iter, self.args.warm_up_iter, self.args.lr)

                loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iter >= self.args.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.optimizer.param_groups[0]['lr']

                if iter % self.args.print_iter == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.writer.add_scalar('Train/%s'%tag, value / self.args.print_iter, iter)
                        mean_loss[tag] = value / self.args.print_iter
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, iter, max_iter, mean_loss, epoch=epoch, inner_iter=i)

                if iter % self.args.save_iter == 0:
                    self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            self.logger.info(f'Validating/Testing: {epoch}')
            self.vq_model.eval()

            val_loss = []
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_perpexity = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_rec, loss_vel, loss_commit, perplexity = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())

            self.writer.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
            self.writer.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
            self.writer.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            self.writer.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
            self.writer.add_scalar('Val/perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)

            self.logger.info('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                  (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                   sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))

            self.logger.info(f'Evaluation: {epoch}')
            best_metrics = evaluation(
                eval_val_loader, eval_wrapper, 'vq_model', self.vq_model,
                logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
                out_dir=self.args.out_dir)


class MaskTransformerTrainer:
    def __init__(self, args, t2m_trans, vq_model, text_model):
        self.args = args
        self.t2m_trans = t2m_trans
        self.vq_model = vq_model
        self.text_model = text_model

        self.device = args.device
            
        self.logger = new_utils.get_logger(args.out_dir)
        self.writer = SummaryWriter(args.out_dir)

# ************************************************************
    def save(self, file_name, epoch, iter):
        state = {
            't2m_trans': self.t2m_trans.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'epoch': epoch,
            'iter': iter,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location='cpu')
        self.t2m_trans.load_state_dict(checkpoint['t2m_trans'], strict=False)

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('Resume wo optimizer or scheduler.')

        epoch, iter = checkpoint['epoch'], checkpoint['iter']
        
        checkpoint.clear()  
        del checkpoint
        torch.cuda.empty_cache()

        return epoch, iter

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
# ************************************************************

    def forward(self, batch_data):
        cond, motion, m_lens, llm_captions, motion_seg = batch_data
        motion_seg = motion_seg.detach().long().to(self.device)

        cond = cond.to(self.device).float() if torch.is_tensor(cond) else cond

        with torch.no_grad():
            cond, action_mask = new_utils.get_action_emb(cond, llm_captions, self.text_model)

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        m_lens = m_lens // 4
        
        code_idx, _ = self.vq_model.encode(motion)
        return_dict = self.t2m_trans(code_idx[..., 0], cond, m_lens, action_mask, motion_seg)

        ce_loss = return_dict['ce_loss']
        align_loss = self.args.align_w * return_dict['align_loss']
        
        acc = return_dict['acc']

        loss = ce_loss + align_loss
        
        losses = {
            'loss': loss,
            'ce_loss': ce_loss,
            'align_loss': align_loss,
        }

        return losses, acc

    def update(self, batch_data):
        losses, acc = self.forward(batch_data)
        loss = losses['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        loss_items = {}
        for loss_type in losses.keys():
            loss_items[loss_type] = losses[loss_type].item()

        return loss_items, acc

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.train_loader = train_loader
        
        self.t2m_trans.to(self.device)
        self.vq_model.to(self.device)

        self.optimizer = optim.AdamW(
            self.t2m_trans.parameters(), 
            betas=(0.9, 0.99), 
            lr=self.args.lr, 
            weight_decay=1e-5)
    
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.args.milestones,
            gamma=self.args.gamma)

        start_time = time.time()
        max_iter = self.args.max_epoch * len(train_loader)
        self.logger.info(f'Total Epochs: {self.args.max_epoch}, Total Iters: {max_iter}')
        self.logger.info('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(val_loader)))

        epoch = 0
        iter = 0
        if self.args.resume_ckpt != '':
            epoch, iter = self.resume(pjoin(self.args.out_dir, 'last.tar'))
            print("Load model epoch:%d iterations:%d"%(epoch, iter))

        logs = defaultdict(def_value, OrderedDict())
        best_metrics = {
            'fid': 100,
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'matching': 100,
            'div': 100,
            'mm': 0
        }
        best_acc = 0
        
        self.logger.info('First evaluation:')
        best_metrics = evaluation(
            eval_val_loader, eval_wrapper, 't2m_trans', self.vq_model,
            text_model=self.text_model,
            t2m_trans=self.t2m_trans, 
            logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
            out_dir=self.args.out_dir,
            time_steps=self.args.time_steps,
            t2m_tem=self.args.t2m_tem, t2m_cond_scale=self.args.t2m_cond_scale)

        while epoch < self.args.max_epoch:
            epoch += 1

            #################################################################
            self.logger.info(f'Training: {epoch}')
            self.t2m_trans.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                iter += 1
                if iter < self.args.warm_up_iter:
                    self.update_lr_warm_up(iter, self.args.warm_up_iter, self.args.lr)
                    
                losses, acc = self.update(batch_data=batch)
                
                for loss_type in losses.keys():
                    logs[loss_type] += losses[loss_type]
                logs['acc'] += acc
                logs['lr'] += self.optimizer.param_groups[0]['lr']

                if iter % self.args.print_iter == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.writer.add_scalar('Train/%s'%tag, value / self.args.print_iter, iter)
                        mean_loss[tag] = value / self.args.print_iter
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, iter, max_iter, mean_loss, epoch=epoch, inner_iter=i)

                if iter % self.args.save_iter == 0:
                    self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            #################################################################
            self.logger.info(f'Validating/Testing: {epoch}')
            self.t2m_trans.eval()
            self.vq_model.eval()
        
            val_losses_dict = defaultdict(list)
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    losses, acc = self.forward(batch_data)

                    for loss_type in losses.keys():
                        val_losses_dict[loss_type].append(losses[loss_type].item())
                    val_acc.append(acc)

            self.logger.info(f"Validation loss:{np.mean(val_losses_dict['loss']):.3f}, accuracy:{np.mean(val_acc):.3f}")

            for loss_type in val_losses_dict.keys():
                self.writer.add_scalar(f'Val/{loss_type}', np.mean(val_losses_dict[loss_type]), epoch)
            self.writer.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                self.logger.info(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}")
                best_acc = np.mean(val_acc)

            #################################################################
            self.logger.info(f'Evaluation: {epoch}')
            best_metrics = evaluation(
                eval_val_loader, eval_wrapper, 't2m_trans', self.vq_model,
                text_model=self.text_model, 
                t2m_trans=self.t2m_trans, 
                logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
                out_dir=self.args.out_dir,
                time_steps=self.args.time_steps,
                t2m_tem=self.args.t2m_tem, t2m_cond_scale=self.args.t2m_cond_scale)


class ResidualTransformerTrainer:
    def __init__(self, args, t2m_trans, res_trans, vq_model, text_model):
        self.args = args
        self.t2m_trans = t2m_trans
        self.res_trans = res_trans
        self.vq_model = vq_model
        self.text_model = text_model

        self.device = args.device
            
        self.logger = new_utils.get_logger(args.out_dir)
        self.writer = SummaryWriter(args.out_dir)
    
# ************************************************************    
    def save(self, file_name, epoch, iter):
        state = {
            'res_trans': self.res_trans.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'epoch': epoch,
            'iter': iter,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location='cpu')
        self.res_trans.load_state_dict(checkpoint['res_trans'], strict=False)

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('Resume wo optimizer or scheduler.')

        epoch, iter = checkpoint['epoch'], checkpoint['iter']
        
        checkpoint.clear()  
        del checkpoint
        torch.cuda.empty_cache()

        return epoch, iter

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
# ************************************************************

    def forward(self, batch_data):
        cond, motion, m_lens, llm_captions, motion_seg = batch_data

        cond = cond.to(self.device).float() if torch.is_tensor(cond) else cond
        
        with torch.no_grad():
            cond, action_mask = new_utils.get_action_emb(cond, llm_captions, self.text_model)

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        m_lens = m_lens // 4
        
        code_idx, all_codes = self.vq_model.encode(motion)
        ce_loss, pred_ids, acc, active_q_layers = self.res_trans(code_idx, cond, m_lens, action_mask)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), acc

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper):
        self.res_trans.to(self.device)
        self.vq_model.to(self.device)

        self.optimizer = optim.AdamW(
            self.res_trans.parameters(), 
            betas=(0.9, 0.99), 
            lr=self.args.lr, 
            weight_decay=1e-5)
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.args.milestones,
            gamma=self.args.gamma)

        start_time = time.time()
        max_iter = self.args.max_epoch * len(train_loader)
        self.logger.info(f'Total Epochs: {self.args.max_epoch}, Total Iters: {max_iter}')
        self.logger.info('Iters Per Epoch, Training: %d, Validation: %d' % (len(train_loader), len(val_loader)))

        epoch = 0
        iter = 0
        if self.args.resume_ckpt != '':
            epoch, iter = self.resume(pjoin(self.args.out_dir, 'last.tar'))
            print("Load model epoch:%d iterations:%d"%(epoch, iter))

        logs = defaultdict(def_value, OrderedDict())
        
        best_metrics = {
            'fid': 100,
            'top1': 0,
            'top2': 0,
            'top3': 0,
            'matching': 100,
            'div': 100,
            'mm': 0
        }

        self.logger.info('First evaluation:')
        model_type = 'all' if self.t2m_trans else 'res_trans'
        best_metrics = evaluation(
            eval_val_loader, eval_wrapper, model_type, self.vq_model,
            text_model=self.text_model, 
            t2m_trans=self.t2m_trans, res_trans=self.res_trans, 
            logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
            out_dir=self.args.out_dir,
            time_steps=self.args.time_steps,
            t2m_tem=self.args.t2m_tem, t2m_cond_scale=self.args.t2m_cond_scale,
            res_tem=self.args.res_tem, res_cond_scale=self.args.res_cond_scale)

        best_loss = 100
        best_acc = 0

        while epoch < self.args.max_epoch:
            epoch += 1
            self.logger.info(f'Training: {epoch}')
            self.res_trans.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                iter += 1
                if iter < self.args.warm_up_iter:
                    self.update_lr_warm_up(iter, self.args.warm_up_iter, self.args.lr)

                loss, acc = self.update(batch_data=batch)
                
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.optimizer.param_groups[0]['lr']

                if iter % self.args.print_iter == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.writer.add_scalar('Train/%s'%tag, value / self.args.print_iter, iter)
                        mean_loss[tag] = value / self.args.print_iter
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, iter, max_iter, mean_loss, epoch=epoch, inner_iter=i)

                if iter % self.args.save_iter == 0:
                    self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            self.save(pjoin(self.args.out_dir, 'last.tar'), epoch, iter)

            self.logger.info(f'Validating/Testing: {epoch}')
            self.res_trans.eval()
            self.vq_model.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            self.logger.info(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.writer.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.writer.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                self.logger.info(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}")
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                self.logger.info(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}")
                best_acc = np.mean(val_acc)

            self.logger.info(f'Evaluation: {epoch}')
            best_metrics = evaluation(
                eval_val_loader, eval_wrapper, model_type, self.vq_model,
                text_model=self.text_model, 
                t2m_trans=self.t2m_trans, res_trans=self.res_trans,
                logger=self.logger, writer=self.writer, epoch=epoch, best_metrics=best_metrics, 
                out_dir=self.args.out_dir,
                time_steps=self.args.time_steps,
                t2m_tem=self.args.t2m_tem, t2m_cond_scale=self.args.t2m_cond_scale,
                res_tem=self.args.res_tem, res_cond_scale=self.args.res_cond_scale)
