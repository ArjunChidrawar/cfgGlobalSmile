import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
import optuna
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class NCLG():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 2:
            model_name = 'inpaint'


        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)


        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        #train mode
        if self.config.MODE == 1:
            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_INPAINT_LANDMARK_FLIST,
                                             config.TRAIN_MASK_FLIST, augment=True, training=True)
                self.val_dataset = Dataset(config, config.VAL_INPAINT_IMAGE_FLIST, config.VAL_INPAINT_LANDMARK_FLIST,
                                           config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        #optuna mode
        elif self.config.MODE == 5:

            if self.config.MODEL == 2:

                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_INPAINT_LANDMARK_FLIST,
                                             config.TRAIN_MASK_FLIST, augment=True, training=True)
                self.val_dataset = Dataset(config, config.VAL_INPAINT_IMAGE_FLIST, config.VAL_INPAINT_LANDMARK_FLIST,
                                           config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)


        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_INPAINT_LANDMARK_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join('/results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def ssim_metric(self, img1, img2):
        # img1, img2: torch tensors, shape (B, C, H, W) or (B, H, W, C)
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
        # Ensure batch dimension
        if img1.ndim == 3:
            img1 = np.expand_dims(img1, 0)
            img2 = np.expand_dims(img2, 0)
        # Convert to (B, H, W, C) if needed
        if img1.shape[1] == 3:
            img1 = np.transpose(img1, (0, 2, 3, 1))
            img2 = np.transpose(img2, (0, 2, 3, 1))
        ssim_vals = [ssim(im1, im2, channel_axis=-1, data_range=255) for im1, im2 in zip(img1, img2)]
        return float(np.mean(ssim_vals))

    def train(self, k_folds=1, patience=5):
        # If k_folds==1, do normal training. If >1, do k-fold cross-validation.
        dataset = self.train_dataset
        indices = np.arange(len(dataset))
        fold_results = []
        if k_folds > 1:
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
                print(f"\n===== Fold {fold+1}/{k_folds} =====")
                train_subset = Subset(dataset, list(train_idx))
                val_subset = Subset(dataset, list(val_idx))
                train_loader = DataLoader(
                    dataset=train_subset,
                    batch_size=self.config.BATCH_SIZE,
                    num_workers=4,
                    drop_last=True,
                    shuffle=True
                )
                if k_folds > 1:
                    val_loader = DataLoader(
                        dataset=val_subset,
                        batch_size=self.config.BATCH_SIZE,
                        num_workers=2,
                        drop_last=False,
                        shuffle=False
                    )
                else:
                    val_loader = DataLoader(
                        dataset=val_subset,
                        batch_size=self.config.BATCH_SIZE,
                        num_workers=2,
                        drop_last=False,
                        shuffle=False
                    )
                best_val_loss = float('inf')
                best_epoch = 0
                epochs_no_improve = 0
                history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
                epoch = 0
                keep_training = True
                model = self.config.MODEL
                max_iteration = int(float((self.config.MAX_ITERS)))
                total = len(train_subset)
                while keep_training:
                    epoch += 1
                    print(f'\n\nTraining epoch: {epoch}')
                    progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
                    train_loss_epoch = 0
                    for items in train_loader:
                        self.inpaint_model.train()
                        if model == 2:
                            images, landmarks, masks = self.cuda(*items)
                        landmarks[landmarks >= self.config.INPUT_SIZE] = self.config.INPUT_SIZE - 1
                        landmarks[landmarks < 0] = 0
                        if model == 2:
                            landmarks[landmarks>=self.config.INPUT_SIZE] = self.config.INPUT_SIZE-1
                            landmarks[landmarks<0] = 0
                            outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(images,landmarks,masks)
                            outputs_merged = (outputs_img * masks) + (images * (1-masks))
                            psnr_val = self.psnr(self.postprocess(images), self.postprocess(outputs_merged)).item()
                            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float().item()
                            ssim_val = self.ssim_metric(self.postprocess(images), self.postprocess(outputs_merged))
                            logs.append(('psnr', psnr_val))
                            logs.append(('mae', mae))
                            logs.append(('ssim', ssim_val))
                            self.inpaint_model.backward(gen_loss, dis_loss)
                            iteration = self.inpaint_model.iteration
                            train_loss_epoch += gen_loss.item()
                        logs = [
                            ("epoch", epoch),
                            ("iter", iteration),
                        ] + logs
                        progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                        if iteration >= max_iteration:
                            keep_training = False
                            break
                    train_loss_epoch /= len(train_loader)
                    history['train_loss'].append(train_loss_epoch)
                    # Validation at end of epoch
                    val_loss, val_psnr, val_ssim = self.validate(val_loader)
                    history['val_loss'].append(val_loss)
                    history['val_psnr'].append(val_psnr)
                    history['val_ssim'].append(val_ssim)
                    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_psnr={val_psnr:.4f}, val_ssim={val_ssim:.4f}")
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        epochs_no_improve = 0
                        self.save()  # Save best model
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                        break
                fold_results.append(history)
        else:
            for fold, (train_idx, val_idx) in enumerate([(indices, indices)]):
                print(f"\n===== Fold {fold+1}/{k_folds} =====")
                train_subset = Subset(dataset, list(train_idx))
                val_subset = self.val_dataset
                train_loader = DataLoader(
                    dataset=train_subset,
                    batch_size=self.config.BATCH_SIZE,
                    num_workers=4,
                    drop_last=True,
                    shuffle=True
                )
                if k_folds > 1:
                    val_loader = DataLoader(
                        dataset=val_subset,
                        batch_size=self.config.BATCH_SIZE,
                        num_workers=2,
                        drop_last=False,
                        shuffle=False
                    )
                else:
                    val_loader = DataLoader(
                        dataset=val_subset,
                        batch_size=self.config.BATCH_SIZE,
                        num_workers=2,
                        drop_last=False,
                        shuffle=False
                    )
                best_val_loss = float('inf')
                best_epoch = 0
                epochs_no_improve = 0
                history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
                epoch = 0
                keep_training = True
                model = self.config.MODEL
                max_iteration = int(float((self.config.MAX_ITERS)))
                total = len(train_subset)
                while keep_training:
                    epoch += 1
                    print(f'\n\nTraining epoch: {epoch}')
                    progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
                    train_loss_epoch = 0
                    for items in train_loader:
                        self.inpaint_model.train()
                        if model == 2:
                            images, landmarks, masks = self.cuda(*items)
                        landmarks[landmarks >= self.config.INPUT_SIZE] = self.config.INPUT_SIZE - 1
                        landmarks[landmarks < 0] = 0
                        if model == 2:
                            landmarks[landmarks>=self.config.INPUT_SIZE] = self.config.INPUT_SIZE-1
                            landmarks[landmarks<0] = 0
                            outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(images,landmarks,masks)
                            outputs_merged = (outputs_img * masks) + (images * (1-masks))
                            psnr_val = self.psnr(self.postprocess(images), self.postprocess(outputs_merged)).item()
                            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float().item()
                            ssim_val = self.ssim_metric(self.postprocess(images), self.postprocess(outputs_merged))
                            logs.append(('psnr', psnr_val))
                            logs.append(('mae', mae))
                            logs.append(('ssim', ssim_val))
                            self.inpaint_model.backward(gen_loss, dis_loss)
                            iteration = self.inpaint_model.iteration
                            train_loss_epoch += gen_loss.item()
                        logs = [
                            ("epoch", epoch),
                            ("iter", iteration),
                        ] + logs
                        progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                        if iteration >= max_iteration:
                            keep_training = False
                            break
                    train_loss_epoch /= len(train_loader)
                    history['train_loss'].append(train_loss_epoch)
                    # Validation at end of epoch
                    val_loss, val_psnr, val_ssim = self.validate(val_loader)
                    history['val_loss'].append(val_loss)
                    history['val_psnr'].append(val_psnr)
                    history['val_ssim'].append(val_ssim)
                    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_psnr={val_psnr:.4f}, val_ssim={val_ssim:.4f}")
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        epochs_no_improve = 0
                        self.save()  # Save best model
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                        break
                fold_results.append(history)
        print('\nK-Fold Cross Validation Results:')
        for i, hist in enumerate(fold_results):
            print(f"Fold {i+1}: Best val_loss={min(hist['val_loss']):.4f}, Best val_psnr={max(hist['val_psnr']):.4f}, Best val_ssim={max(hist['val_ssim']):.4f}")
        return fold_results

    def validate(self, val_loader):
        self.inpaint_model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        count = 0
        with torch.no_grad():
            for items in val_loader:
                images, landmarks, masks = self.cuda(*items)
                outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(images, landmarks, masks)
                outputs_merged = (outputs_img * masks) + (images * (1-masks))
                loss = gen_loss.item()
                psnr_val = self.psnr(self.postprocess(images), self.postprocess(outputs_merged)).item()
                ssim_val = self.ssim_metric(self.postprocess(images), self.postprocess(outputs_merged))
                total_loss += loss
                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1
        return total_loss / count, total_psnr / count, total_ssim / count

    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        print('here')
        index = 0
        for items in test_loader:
            images, landmarks, masks = self.cuda(*items)
            index += 1

            if model == 2:
                landmarks[landmarks >= self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
                landmarks[landmarks < 0] = 0

                inputs = (images * (1 - masks))

                outputs_img, outputs_lmk = self.inpaint_model(images, masks)
                outputs_lmk *= self.config.INPUT_SIZE
                outputs_lmk = outputs_lmk.reshape((-1, self.config.LANDMARK_POINTS, 2))

                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked')
                path_result = os.path.join(self.results_path, self.model_name,'result')
                path_joint = os.path.join(self.results_path,self.model_name,'joint')

                path_landmark_output = os.path.join(self.results_path, self.model_name, 'landmark_output')
                path_landmark_only = os.path.join(self.results_path, self.model_name, 'landmark_only')

                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)

                create_dir(path_landmark_output)
                create_dir(path_landmark_only)


                landmark_output_image = outputs_img
                landmark_output_image = (landmark_output_image.squeeze().cpu().detach().numpy().transpose(1,2,0)*255).astype('uint8')
                landmark_output_image = landmark_output_image.copy()
                for i in range(outputs_lmk.shape[1]):
                    circle(landmark_output_image, (int(outputs_lmk[0, i, 0]), int(outputs_lmk[0, i, 1])), radius=2,
                           color=(0, 255, 0), thickness=-1)

                landmark_map = torch.zeros(1,3,self.config.INPUT_SIZE, self.config.INPUT_SIZE)
                landmark_map = (landmark_map.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                landmark_map = np.array(landmark_map)
                landmark_map = landmark_map.copy()

                for i in range(outputs_lmk.shape[1]):
                    circle(landmark_map,(int(outputs_lmk[0, i, 0]), int(outputs_lmk[0, i, 1])), radius=2, color=(0,255, 0), thickness=-1)

                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint,name[:-4]+'.png'))

                landmark_output_image = Image.fromarray(landmark_output_image)
                landmark_output_image.save(os.path.join(path_landmark_output, name))

                landmark_map = Image.fromarray(landmark_map)
                landmark_map.save(os.path.join(path_landmark_only,name))

                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                print(name + ' complete!')

        print('\nEnd Testing')



    def sample(self, it=None):
        self.inpaint_model.eval()

        model = self.config.MODEL

        items = next(self.sample_iterator)

        if model == 2:
            images,landmarks,masks = self.cuda(*items)

        landmarks[landmarks>=self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
        landmarks[landmarks<0] = 0


        # inpaint model
        if model == 2:


            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            for i in range(inputs.shape[0]):
                inputs[i, :, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1-masks[i,0,landmarks[i, :, 1], landmarks[i,:,0]]

            outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(
                images, landmarks, masks)
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))


        if it is not None:
            iteration = it


        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1


        elif model == 2:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs_img),
                self.postprocess(outputs_merged),
                img_per_row=image_per_row
            )



        if iteration % 200 == 0:
            path = os.path.join(self.samples_path, self.model_name)
            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images.save(name)

        return outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):

        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
