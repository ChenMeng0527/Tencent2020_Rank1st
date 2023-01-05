'''
预训练bert模型
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
# from transformers.modeling_bert import BertLayerNorm

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


        
class Model(nn.Module):   
    def __init__(self, encoder, config, args):
        '''
        encoder：transformers中的模型
        '''
        super(Model, self).__init__()
        self.encoder = encoder
        self.lm_head = []

        # embdding并初始化 [vocab_size_v1,vocab_dim_v1] = [15xxxxx, 64]
        # args.vocab_size_v1：所有id的索引
        # args.vocab_dim_v1：64

        self.text_embeddings = nn.Embedding(args.vocab_size_v1, args.vocab_dim_v1)
        self.text_embeddings.apply(self._init_weights)

        # NN 并初始化
        # args.text_dim：8*128
        # args.vocab_dim_v1：64
        # len(args.text_features)：8
        # [8*128 + 64*8, 8*128]
        self.text_linear = nn.Linear(args.text_dim + args.vocab_dim_v1 * len(args.text_features), config.hidden_size)
        self.text_linear.apply(self._init_weights)

        # args.vocab_size: 所有id的尺寸[x,x,x,x,x,x,x,x]
        for x in args.vocab_size:
            # [8*128,x]
            self.lm_head.append(nn.Linear(config.hidden_size, x, bias=False))
        self.lm_head = nn.ModuleList(self.lm_head)
        self.config = config
        self.args = args


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, inputs, inputs_ids, masks, labels):
        '''
            # 1: text_features : [block_size,text_dim]  [B,S,8*128]
            # 2: text_ids : [block_size,len(self.args.text_features)]  [B,S,8]
            # 3: text_masks : [block_size]  [B,S]
            # 4: text_label : [block_size,len(self.args.text_features)]  [B,S,8]

        '''

        # -----------1:将输入的id转为embedding,然后与本身的预训练embedding进行拼接，然后全联接+relu-----------
        # [S,8,E]
        # inputs_ids: [B,S,8] ---text_embeddings = [B,S,8,64] --view = [B,S,8*64]
        inputs_embedding = self.text_embeddings(inputs_ids).view(inputs.size(0), inputs.size(1), -1)
        # inputs = [B,S,8*128]
        # inputs = [B,S,8*128]+[B,S,8*64] = [B,S,8*128+8*64]
        inputs = torch.cat((inputs.float(), inputs_embedding), -1)
        # [B,S,8*128+8*64] -- [B,S,hidden_size]
        inputs = torch.relu(self.text_linear(inputs))


        # -----------2:
        # bert模型输入: embedding + masks作为输入
        # embedding:[B,S,hidden_size]
        # masks:[B,S]
        outputs = self.encoder(inputs_embeds=inputs, attention_mask=masks.float())[0]
        loss = 0
        #
        for idx, (x, y) in enumerate(zip(self.lm_head, self.args.text_features)):
            if y[3] is True:
                # labels[:, :, idx]不等于-100？ 默认值为-100，即如果改变了就为true，没变化为false
                # labels[:, :, idx]后，将labels [B,S,8]变为 [B,S]
                outputs_tmp = outputs[labels[:, :, idx].ne(-100)]
                # 每种行为下的label[B,S]
                labels_tmp = labels[:, :, idx]
                # 转为01
                labels_tmp = labels_tmp[labels_tmp.ne(-100)].long()
                # 将[B,S]经过全联接wx+b
                prediction_scores = x(outputs_tmp)
                loss_fct = CrossEntropyLoss()
                # [B,S]log 与 [B,S] 0/1做交叉墒
                # 最终 tensor(0.6931)
                masked_lm_loss = loss_fct(prediction_scores, labels_tmp)  
                loss = loss + masked_lm_loss
        return loss


#
outputs = torch.tensor([[0,1],
                        [1,0]]).float()
label = torch.tensor([[0,1],
                      [1,0]]).float()
loss_fct = CrossEntropyLoss()
masked_lm_loss = loss_fct(outputs, label)

