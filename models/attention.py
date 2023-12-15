import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(32, 50, 256)
         >>> context = torch.randn(32, 1, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([32, 50, 256])
         >>> weights.size()
         torch.Size([32, 50, 1])
    """

    def __init__(self, dimensions):
        super(Attention, self).__init__()

        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, attention_mask):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.
            output length: length of utterance
            query length: length of each token (1)
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 2)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        attention_weights = self.softmax(attention_scores)
        mix = torch.bmm(attention_weights, context)
        combined = torch.cat((mix, query), dim=2)
        output = self.linear_out(combined)

        output = self.tanh(output)
        return output, attention_weights

class LabelAttention(nn.Module):

    def __init__(self, size, d_a, n_labels):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(LabelAttention, self).__init__()
        self.size = size
        self.n_labels = n_labels
        self.d_a = d_a
        
        self.first_linears = nn.Linear(self.size, self.d_a, bias=False)
        self.second_linears = nn.Linear(self.d_a, self.n_labels, bias=False)
        self.third_linears = nn.Linear(self.size, self.n_labels, bias=True)

        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        torch.nn.init.normal(self.first_linears.weight, mean, std)
        if self.first_linears.bias is not None:
            self.first_linears.bias.data.fill_(0)

    def forward(self, x):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        weights = torch.tanh(self.first_linears(x))

        att_weights = self.second_linears(weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        context_vector = att_weights @ x

        logits = self.third_linears.weight.mul(context_vector).sum(dim=2).add(self.third_linears.bias)

        return context_vector, logits

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)
    
class SlotToIntent(nn.Module):
    def __init__(self, dim_x, dim_y, out_dim, dropout_rate=0.):
        super(SlotToIntent, self).__init__()

        self.linear_x = nn.Linear(dim_x, out_dim, bias=True)
        self.linear_y = nn.Linear(dim_y, out_dim, bias=True) 

        self.a_x = nn.Linear(out_dim, 1)
        self.a_y = nn.Linear(out_dim, 1)

        self.w = nn.Parameter(
            torch.randn(dim_x, dim_y),
            requires_grad=True
            )

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # batch_size x n x l
        C = torch.tanh(torch.einsum('bnx,xy,bly->bnl', x, self.w, y))
        # batch_size x n x d
        proj_x = self.linear_x(x)
        # batch_size x l x d
        proj_y = self.linear_y(y)

        # batch_size x n x d
        H_x = torch.tanh(proj_x + torch.bmm(C, proj_y))
        # batch_size x n
        a_x = self.softmax(self.a_x(H_x).squeeze(2))
        X = torch.einsum('bnd,bn->bd', x, a_x)
        return X