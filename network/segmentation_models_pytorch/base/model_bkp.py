import torch
import torch.nn as nn
from . import initialization as init

import torch.nn.functional as F

import matplotlib.pyplot as plt
fig = plt.figure()
global zi
zi = 0

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

#### modifications for rgb/thermal segmentation    

#### double encoder
class SegmentationModelDoubleEncoder(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, debug_viz=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        x_1 = x[:,:3] # rgb image
        x_2 = x[:,3].unsqueeze(1).repeat(1,3,1,1) # thermal image

        # concatenate deep features of two encoders
        features_1 = self.encoder_1(x_1)
        features_2 = self.encoder_2(x_2)

        features = []

        b = features_1[0].shape[0]

        #### make a rough prediction with only thermal or only rgb
        
        if self.triple_decoder:
            #### using two new decoders:
            decoder_output_1 = self.decoder_1(*features_1)
            masks_1 = self.segmentation_head_1(decoder_output_1)
            confidence_weight_1, _ = torch.max(torch.softmax(masks_1, dim=1), dim=1)
            confidence_weight_1 = confidence_weight_1.unsqueeze(1)
            confidence_weight_1_sum = torch.sum(confidence_weight_1)

            decoder_output_2 = self.decoder_2(*features_2)
            masks_2 = self.segmentation_head_2(decoder_output_2)
            confidence_weight_2, _ = torch.max(torch.softmax(masks_2, dim=1), dim=1)
            confidence_weight_2 = confidence_weight_2.unsqueeze(1)
            confidence_weight_2_sum = torch.sum(confidence_weight_1)

        else:
            confidence_weight_1 = 1
            confidence_weight_2 = 1
            confidence_weight_1_shallow = 1
            confidence_weight_2_shallow = 1
            confidence_weight_1_deep = 1
            confidence_weight_2_deep = 1
        
        # correlation between confidences
        if self.correlation_weight:
            masks_1_deep = F.interpolate(masks_1, size=features_2[-1].shape[2:], mode='bilinear', align_corners=True)
            masks_2_deep = F.interpolate(masks_2, size=features_2[-1].shape[2:], mode='bilinear', align_corners=True)

            feature_corr_deep_1 = self.fc(torch.softmax(masks_1_deep,dim=1), torch.softmax(masks_2_deep,dim=1))

            theta_deep = self.fw_deep(feature_corr_deep_1)
            theta_shallow = F.interpolate(theta_deep, size=features_2[-4].shape[2:], mode='bilinear', align_corners=True)
            theta_deep = F.interpolate(theta_deep, size=features_2[-1].shape[2:], mode='bilinear', align_corners=True)
            
        else:
            theta_deep = 1
            theta_shallow = 1

        # concatenate third layer feats to retrieve high resolution details
        features += features_1[:-4] + [theta_shallow * torch.cat((confidence_weight_1 * features_1[-4], confidence_weight_2 * features_2[-4]), 1)]

        if self.triple_decoder:
            confidence_weight_1_deep = F.interpolate(confidence_weight_1, size=features_2[-1].shape[2:], mode='bilinear', align_corners=True)
            confidence_weight_2_deep = F.interpolate(confidence_weight_2, size=features_2[-1].shape[2:], mode='bilinear', align_corners=True)

        start_index = -3
        #start_index = 0
        features += features_1[start_index:-1] + [theta_deep * torch.cat((confidence_weight_1_deep * features_1[-1], confidence_weight_2_deep * features_2[-1]), 1)] # concatenation with confidence and correlation_weight

        decoder_output = self.decoder(*features)

        # fusion of high resolution images
        concat_first_layer = False
        if concat_first_layer:
            x_fused = x
            x_fused = self.image_fusion(x)
            # upsample decoder_output
            decoder_output = self.decoder_upsampling(decoder_output)
            # concatenate 
            decoder_output = torch.cat((decoder_output, x_fused), 1)
            

        masks = self.segmentation_head(decoder_output)

        if debug_viz:
            features_1_img = torch.mean(features_1[-4], axis=1)
            features_1_img_np = features_1_img[0].detach().cpu().numpy()
            features_2_img = torch.mean(features_2[-4], axis=1)
            features_2_img_np = features_2_img[0].detach().cpu().numpy()
            features_1_img_deep = torch.mean(features_1[-1], axis=1)
            features_1_img_np_deep = features_1_img_deep[0].detach().cpu().numpy()
            features_2_img_deep = torch.mean(features_2[-1], axis=1)
            features_2_img_np_deep = features_2_img_deep[0].detach().cpu().numpy()

            features_1_img_deep_w = torch.mean(theta_deep*confidence_weight_1_deep*features_1[-1], axis=1)
            features_1_img_np_deep_w = features_1_img_deep_w[0].detach().cpu().numpy()
            features_2_img_deep_w = torch.mean(theta_deep*confidence_weight_1_deep*features_2[-1], axis=1)
            features_2_img_np_deep_w = features_2_img_deep_w[0].detach().cpu().numpy()

            plt.subplot(3,2,1)
            plt.imshow(features_1[0][0][:3].permute(1,2,0).detach().cpu().numpy())
            plt.subplot(3,2,2)
            plt.imshow(features_2[0][0][0].detach().cpu().numpy(),cmap='gray')

            plt.subplot(3,2,3)
            plt.imshow(features_1_img_np_deep,)
            plt.subplot(3,2,4)
            plt.imshow(features_2_img_np_deep,)
            plt.subplot(3,2,5)

            # plt.imshow(theta_deep[0][0].detach().cpu().numpy()) # plt.imshow(features_1_img_np,)
            # plt.subplot(3,2,4)
            # plt.imshow(confidence_weight_2[0][0].detach().cpu().numpy())
            # plt.subplot(3,2,5)

            plt.imshow(features_1_img_np_deep_w) # plt.imshow(features_1_img_np,)
            plt.subplot(3,2,6)
            plt.imshow(features_2_img_np_deep_w)

            #plt.show()
            #plt.pause(.0000001) # Delay in seconds
            #fig.canvas.draw() # Draws the image to the screen
            #plt.close()

            global zi
            plt.savefig(f'/home/ofrigo/debug_improved{zi:05}.png')
            zi+=1
        
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
    

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
