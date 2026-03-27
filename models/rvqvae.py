from torch import nn

from models.encdec import Encoder, Decoder
from models.quantizer import ResidualVQ


class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        motion_dim = 251 if args.dataname == 'kit' else 263
        
        self.encoder = Encoder(motion_dim, 
                               code_dim, 
                               down_t, stride_t, 
                               width, depth,
                               dilation_growth_rate, 
                               activation=activation, norm=norm)
        
        self.decoder = Decoder(motion_dim, 
                               code_dim, 
                               down_t, 
                               width, depth,
                               dilation_growth_rate, 
                               activation=activation, norm=norm)
        
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        
        x_encoder = self.encoder(x_in)
        
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        
        return code_idx, all_codes

    def forward(self, x):
        x_in = self.preprocess(x)
        
        x_encoder = self.encoder(x_in)
        
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        x_decoder = self.decoder(x_quantized)
        
        x_out = self.postprocess(x_decoder)

        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)

        x = x_d.sum(dim=0).permute(0, 2, 1)

        x_decoder = self.decoder(x)
        
        x_out = self.postprocess(x_decoder)

        return x_out
