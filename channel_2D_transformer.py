import torch
import torch.nn as nn

'''
The Channel2DTransformer module was proposed by the paper: "Channel2DTransformer: a Multi-level Features Self-Attention Fusion Module For Semantic Segmentation"
'''
class Channel2DTransformer(nn.Module):
    def __init__(self, inChannle, numObjectFeature, qkvChannle=None, shareQkvConv=False,qkvChannleUseGroupGenerate=True,qkv_kernalSize=1,useScaleFor_q=True,doAttention=True):
        super(Channel2DTransformer, self).__init__()

        self.numObjectFeature = numObjectFeature
        self.qkvChannle = qkvChannle if qkvChannle else inChannle
        self.shareQkvConv=shareQkvConv
        self.qkvChannleUseGroupGenerate=qkvChannleUseGroupGenerate
        self.doAttention=doAttention
        self.useScaleFor_q=useScaleFor_q

        groups=1
        qkvMultiCoff=3 if self.doAttention else 1

        if self.qkvChannleUseGroupGenerate:
            assert self.qkvChannle<=inChannle,f"qkvChannle:{self.qkvChannle} not allow bigger than inChannle{inChannle}"
            groups*=self.qkvChannle

        if not shareQkvConv:
            groups *= numObjectFeature
            self.qkvConv = nn.Conv2d(
                inChannle * numObjectFeature,
                self.qkvChannle * qkvMultiCoff * numObjectFeature,
                kernel_size=qkv_kernalSize, stride=1, padding=(qkv_kernalSize - 1) // 2, groups=groups,
                bias=False)
        else:
            self.qkvConv = nn.Conv2d(
                inChannle,
                self.qkvChannle * qkvMultiCoff ,
                kernel_size=qkv_kernalSize, stride=1, padding=(qkv_kernalSize - 1) // 2, groups=groups,
                bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def attention(self,object_features):
        channle_concat_feature = torch.cat(object_features, dim=1)  # [B,inChannle * numObjectFeature,H,W]
        if not self.shareQkvConv:
            qkv = self.qkvConv(channle_concat_feature)  # [B,qkvChannle * 3 * numObjectFeature,H,W]
        else:
            B_temp, C_temp, H_temp, W_temp = channle_concat_feature.shape
            channle_concat_feature_view=channle_concat_feature.view(B_temp*self.numObjectFeature,-1,H_temp,W_temp)#[B* numObjectFeature,inChannle,H,W]
            qkv = self.qkvConv(channle_concat_feature_view)  # [B* numObjectFeature,qkvChannle * 3 ,H,W]
            qkv=qkv.view(B_temp,-1,H_temp,W_temp)# [B,qkvChannle * 3 * numObjectFeature,H,W]

        B, C, H, W = qkv.shape

        self.scale = (H*W) ** -0.5 if self.useScaleFor_q else 1

        qkv = qkv.view(B, self.numObjectFeature, -1, H, W)  # [B,numObjectFeature,qkvChannle * 3 ,H,W]
        if self.qkvChannleUseGroupGenerate:
            qkv_group=qkv.view(B, self.numObjectFeature,self.qkvChannle,3, H, W)  # [B,numObjectFeature,qkvChannle , 3 ,H,W]
            q = self.scale * qkv_group[:, :, :, 0, ...]  # [B,numObjectFeature,qkvChannle,H,W]
            k = qkv_group[:, :, :, 1, ...]  # [B,numObjectFeature,qkvChannle,H,W]
            v = qkv_group[:, :, :, 2, ...]  # [B,numObjectFeature,qkvChannle,H,W]
        else:
            q = self.scale * qkv[:, :, 0:self.qkvChannle, ...]  # [B,numObjectFeature,qkvChannle,H,W]
            k = qkv[:, :, self.qkvChannle:self.qkvChannle * 2, ...]  # [B,numObjectFeature,qkvChannle,H,W]
            v = qkv[:, :, self.qkvChannle * 2:self.qkvChannle * 3, ...]  # [B,numObjectFeature,qkvChannle,H,W]

        q = q.permute(1, 0, 2, 3, 4)  # [numObjectFeature,B,qkvChannle ,H,W]
        k = k.permute(0, 2, 3, 4, 1)  # [B,qkvChannle ,H,W,numObjectFeature]
        v = v.permute(0, 2, 3, 4, 1)  # [B,qkvChannle ,H,W,numObjectFeature]

        A = torch.einsum('a b c d e, b c d e f -> a b c d e f', q,
                         k)  # [numObjectFeature,B,qkvChannle ,H,W,numObjectFeature]
        A_h_sum = torch.sum(A, dim=3,keepdim=True)  # [numObjectFeature,B,qkvChannle,1,W,numObjectFeature]
        A_h_w_sum = torch.sum(A_h_sum, dim=4,keepdim=True)  # [numObjectFeature,B,qkvChannle,1,1,numObjectFeature]

        A_h_w_sum_softmax = torch.softmax(A_h_w_sum, dim=-1)  # [numObjectFeature,B,1,1,numObjectFeature]

        v_unsqueeze = torch.unsqueeze(v, dim=0)  # [1,B,qkvChannle,H,W,numObjectFeature]

        F = A_h_w_sum_softmax * v_unsqueeze  # [numObjectFeature,B,qkvChannle,H,W,numObjectFeature]
        result = torch.sum(F, dim=-1)  # [numObjectFeature,B,qkvChannle,H,W]

        return [result[i] for i in range(self.numObjectFeature)]  # [(B,qkvChannle,H,W),(B,qkvChannle,H,W),...]


    def not_attention(self,object_features):
        channle_concat_feature = torch.cat(object_features, dim=1)  # [B,inChannle * numObjectFeature,H,W]
        if not self.shareQkvConv:
            qkv = self.qkvConv(channle_concat_feature)  # [B,qkvChannle * numObjectFeature,H,W]
        else:
            B_temp, C_temp, H_temp, W_temp = channle_concat_feature.shape
            channle_concat_feature_view = channle_concat_feature.view(B_temp * self.numObjectFeature, -1, H_temp,
                                                                      W_temp)  # [B* numObjectFeature,inChannle,H,W]
            qkv = self.qkvConv(channle_concat_feature_view)  # [B* numObjectFeature,qkvChannle ,H,W]
            qkv = qkv.view(B_temp, -1, H_temp, W_temp)  # [B,qkvChannle * numObjectFeature,H,W]


        B, C, H, W = qkv.shape

        qkv = qkv.view(B, self.numObjectFeature, -1, H, W)  # [B,numObjectFeature,qkvChannle,H,W]
        result=qkv.permute(1, 0, 2, 3, 4)# [numObjectFeature,B,qkvChannle,H,W]

        return [result[i] for i in range(self.numObjectFeature)]  # [(B,qkvChannle,H,W),(B,qkvChannle,H,W),...]

    # object_features:[(B,C,H,W),(B,C,H,W),(B,C,H,W),...]
    def forward(self, object_features):
        assert self.numObjectFeature == len(object_features), "self.numObjectFeatureb != len(object_features)"

        if self.doAttention:
            return self.attention(object_features)
        else:
            return self.not_attention(object_features)


if __name__ == "__main__":
    B, C, H, W = 2, 3, 10, 20
    multi_level_features = [torch.rand((B,C,H,W)) for i in range(4)]

    channel_2D_transformer = Channel2DTransformer(C, len(multi_level_features))

    attention_features = channel_2D_transformer(multi_level_features)


    print([atten.shape for atten in attention_features])
