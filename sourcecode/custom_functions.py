from fastai.text import *
import pandas as pd
from seq2seq import DataBunch, Seq2SeqTextList, TeacherForcing

# in develoment library, see https://github.com/facebookresearch/fastText to download, how to build, etc.
import fasttext as ft

path = Path('../data')
in_data_file = path/'processed_data.csv'
databunch_file = path/'databunch.pkl'

custom_batch_size = 64
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class Seq2SeqRNN(nn.Module):
    def __init__(self, emb_enc, emb_dec, 
                    nh, out_sl, 
                    nl=2, bos_idx=0, pad_idx=1):
        super().__init__()
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.bos_idx,self.pad_idx = bos_idx,pad_idx
        self.em_sz_enc = emb_enc.embedding_dim
        self.em_sz_dec = emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings
                 
        self.emb_enc = emb_enc
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(self.em_sz_enc, nh, num_layers=nl,
                              dropout=0.25, batch_first=True)
        self.out_enc = nn.Linear(nh, self.em_sz_dec, bias=False)
        
        self.emb_dec = emb_dec
        self.gru_dec = nn.GRU(self.em_sz_dec, self.em_sz_dec, num_layers=nl,
                              dropout=0.1, batch_first=True)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(self.em_sz_dec, self.voc_sz_dec)
        self.out.weight.data = self.emb_dec.weight.data
        
    def encoder(self, bs, inp):
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        _, h = self.gru_enc(emb, h)
        h = self.out_enc(h)
        return h
    
    def decoder(self, dec_inp, h):
        emb = self.emb_dec(dec_inp).unsqueeze(1)
        outp, h = self.gru_dec(emb, h)
        outp = self.out(self.out_drop(outp[:,0]))
        return h, outp
        
    def forward(self, inp):
        bs, sl = inp.size()
        h = self.encoder(bs, inp)
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        
        res = []
        for i in range(self.out_sl):
            h, outp = self.decoder(dec_inp, h)
            dec_inp = outp.max(1)[1]
            res.append(outp)
            if (dec_inp==self.pad_idx).all(): break
        return torch.stack(res, dim=1)
    
    def initHidden(self, bs): return one_param(self).new_zeros(self.nl, bs, self.nh)

def seq2seq_loss(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    return CrossEntropyFlat()(out, targ)

def seq2seq_acc(out, targ, pad_idx=1):
    bs,targ_len = targ.size()
    _,out_len,vs = out.size()
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    out = out.argmax(2)
    return (out==targ).float().mean()

def create_emb(vecs, itos, em_sz=300, mult=1.):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    vec_dic = {w:vecs.get_word_vector(w) for w in vecs.get_words()}
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = tensor(vec_dic[w])
        except: miss.append(w)
    return emb

def get_emb(data):
    emb_enc = None
    emb_dec = None

    en_vecs = ft.load_model(str((path/'cc.en.300.bin')))

    if not os.path.exists(path/'emb_enc.pth'):
        # creating embedding takes several minutes 5? maybe more
        print('Creating embedding')
        emb_enc = create_emb(en_vecs, data.x.vocab.itos)
        emb_dec = create_emb(en_vecs, data.y.vocab.itos)
        del en_vecs # release memory from vector
        torch.save(emb_enc, path/'emb_enc.pth')
        torch.save(emb_dec, path/'emb_dec.pth')
    else:
        print('Loading embedding')
        emb_enc = torch.load(path/'emb_enc.pth')
        emb_dec = torch.load(path/'emb_dec.pth')
    return emb_enc, emb_dec

def get_learner():
    data = get_data()
    emb_enc, emb_dec = get_emb(data)
    xb,yb = next(iter(data.valid_dl))
    rnn = Seq2SeqRNN(emb_enc, emb_dec, 256, max(xb.shape[1], yb.shape[1]))
    h = rnn.encoder(custom_batch_size, xb.cpu())
    learn = Learner(data, rnn, loss_func=seq2seq_loss, metrics=seq2seq_acc)
    return learn

def get_data():
    df = pd.read_csv(in_data_file, low_memory=False, na_filter=False)
    src = (Seq2SeqTextList.from_df(df, path = path, cols='annotated_text')
       .split_by_rand_pct(seed=seed)
       .label_from_df(cols='orig_text', label_cls=TextList))
    return src.databunch()
