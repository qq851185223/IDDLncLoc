import Bio.SeqIO as Seq

from feature_extracting.CTD.feamodule import CTD
from feature_extracting.CTD.feamodule import ORF_length as len
from feature_extracting.CTD.feamodule import ProtParam as PP
from feature_extracting.CTD.feamodule import fickett


def get_CTD():
    seq_file = '../../data/input.fasta'
    tmp = open('../../data/CTD.txt', 'w')
    feature = open('CTD_tmp.txt', 'w')
    out_label = 1
    feature.write("\t".join(map(str,
                                ["#ID", "ORF-integrity", "ORF-coverage", "Instability", "T2", "C0", "PI", "ORF-length",
                                 "AC", "T0", "G0", "C2", "A4", "G2", "TG", "A0", "TC", "G1", "C3", "T3", "A1", "GC",
                                 "T1", "G4", "C1", "G3", "A3", "Gravy", "Hexamer", "C4", "AG", "Fickett", "A2", "T4",
                                 "C", "G", "A", "T"])) + "\n")
    for seq in Seq.parse(seq_file, 'fasta'):
        seqid = seq.id
        A, T, G, C, AT, AG, AC, TG, TC, GC, A0, A1, A2, A3, A4, T0, T1, T2, T3, T4, G0, G1, G2, G3, G4, C0, C1, C2, C3, C4 = CTD.CTD(
            seq.seq)
        insta_fe, PI_fe, gra_fe = PP.param(seq.seq)
        fickett_fe = fickett.fickett_value(seq.seq)
        # hexamer = FrameKmer.kmer_ratio(seq.seq,6,3,coding,noncoding)
        Len, Cov, inte_fe = len.len_cov(seq.seq)
        tem = [inte_fe, Cov, insta_fe, T2, C0, PI_fe, Len, AC, T0, G0, C2, A4, G2, TG, A0, TC, G1, C3, T3, A1, GC, T1,
               G4, C1, G3, A3, gra_fe, C4, AG, fickett_fe, A2, T4, C, G, A, T]
        feature.write(" ".join(map(str,
                                   [inte_fe, Cov, insta_fe, T2, C0, PI_fe, Len, AC, T0, G0, C2, A4, G2, TG, A0, TC, G1,
                                    C3, T3, A1, GC, T1, G4, C1, G3, A3, gra_fe, C4, AG, fickett_fe, A2, T4, C, G, A,
                                    T])) + "\n")
        # tmp.write('%d'%(out_label)),
        for label, item in enumerate(tem):
            tmp.write(' %.6f' % float(item))
        tmp.write('\n')
    tmp.close()
    print('完成CTD特征的提取')


if __name__ == '__main__':
    get_CTD()
