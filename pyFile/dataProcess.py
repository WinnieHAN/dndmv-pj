import os, sys, torch, codecs
from trainer import *

def main(maintype):
    # 'pascalprocess'
    if maintype=='fast2conll':
        def dep_from_hdpdep_output(dep_in):
            deps_p_c = dep_in[1:]
            deps = {}
            for pc in deps_p_c:
                [p, c] = [int(i) for i in pc.split('-')]
                if p == len(deps_p_c):
                    deps[c] = 0
                else:
                    deps[c] = p + 1
            return deps

        def search(path, word):
            for filename in os.listdir(path):
                fp = os.path.join(path, filename)
                if word in filename:
                    return fp
                elif os.path.isdir(fp):
                    search(fp, word)


        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos'
        instc = 'train_init_conll'
        intdep = '/home/hanwj/hdp_dep/code_'
        outstc = 'train_hdpdep_init_conll'

        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            runfile = search(intdep + lang, 'run__')
            in_dep = runfile+'/out_put'
            out_stc = os.path.join(os.path.join(mainPath, lang), outstc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))

            stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs_dep = [[[k for k in line.rstrip(' ').split(' ')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                token_dep = dep_from_hdpdep_output(token_stcs_dep[i][3]) # read
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                           token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + '\t' + str(token_dep[j]) + '\n'
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='findtags':
        tagdic = {}
        num_tag = 0
        path = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP/'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']

        for lang in langs:
            stcs = codecs.open(path+lang+'/poses_english', 'r', encoding='utf8').read().rstrip('\n').split('\n')
            token_stcs = [[j for j in i.rstrip(' ').split(' ')] for i in stcs]
            for i in token_stcs:
                for j in i:
                    if j not in tagdic:
                        tagdic[j] = num_tag
                        num_tag = num_tag + 1
        print(tagdic)

    elif maintype=='conllforHDPDEP':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        # ['test', 'test-all']  '_upos_conll'
        instc = 'train_upos_conll'  # 'test-all_upos_conll'
        outtags = 'poses_english'
        outwords = 'words_english'
        outdeps = 'deps_english'
        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))

            out_tags = os.path.join(os.path.join(mainPath, lang), outtags)
            out_words = os.path.join(os.path.join(mainPath, lang), outwords)
            out_deps = os.path.join(os.path.join(mainPath, lang), outdeps)

            wf = codecs.open(out_tags, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][4] + ' '
                    wf.write(line)
                wf.write('#\n')
            wf.close()

            wf = codecs.open(out_words, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][1] + ' '
                    wf.write(line)
                wf.write('#\n')
            wf.close()

            wf = codecs.open(out_deps, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    par = 0
                    if j==0:
                        j = len(token_stcs[i])
                    # if int(token_stcs[i][j][6])==0:
                    #     par = len(token_stcs[i])
                    # else:
                    #     par = int(token_stcs[i][j][6]) - 1
                    line = str(par) + '-' + str(j) + ' '
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='pascalinit_utf':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos'
        langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        instc = 'train_init_conll'
        intdep = 'train_ndmv_init_conll'
        outstc = 'train_ndmv_init_conll_new'
        for lang in langs:
            in_stc = os.path.join(os.path.join(mainPath, lang), instc)
            in_dep = os.path.join(os.path.join(mainPath, lang), intdep)
            out_stc = os.path.join(os.path.join(mainPath, lang), outstc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]
            stcs_num = len(token_stcs)
            print(lang + '  num:  ' + str(stcs_num))
            stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                           token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                           '\t' + token_stcs_dep[i][j][6] + '\n'
                    wf.write(line)
                wf.write('\n')
            wf.close()


    elif maintype=='pascalprocess':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data/pascal-pos-forHDPDEP'
        langs = ['arabic', 'basque', 'english', 'childes', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']

        for train_dev_test in ['train']:  # ['train', 'dev', 'test', 'test-all']:
            for lang in langs:
                f_name = os.path.join(os.path.join(mainPath, lang), train_dev_test)
                f_name_w = os.path.join(os.path.join(mainPath, lang), train_dev_test + '_upos_conll')
                stcs = open(f_name).read().rstrip("\n").split("\n\n")

                # temp1_sts = [stc.rstrip("\n").rsplit("\n") for stc in stcs]
                # temp2_sts = [line.rstrip('\t').split('\t') for gg]

                token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs]

                stcs_num = len(token_stcs)
                print(lang + '  num:  ' + str(stcs_num))

                position_word = 0
                position_ftag = 1
                position_utag = 2
                position_parent = 3

                wf = open(f_name_w, 'w')

                for i in range(stcs_num):
                    for j in range(len(token_stcs[i][0])):  # one sentences
                        line = '_'+'\t' + token_stcs[i][position_word][j] + '\t' + '_' + '\t' + '_' + '\t' + token_stcs[i][position_utag][j] + '\t' + '_' + \
                               '\t' +token_stcs[i][position_parent][j] + '\n'
                        wf.write(line)
                    wf.write('\n')

                wf.close()
                # abstractSts2File_dev = "data/forWord2Vec/wsj-inf_2-21_dep_filter_10_abstractSts_dev"
                # sentences_dev = open(os.path.join(nowPath, abstractSts2File_dev)).read().rstrip("\n").split("\n")
                # sentences_words_dev = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_dev if
                #                        len(i.rstrip("\t").split('\t')) < 11]
                # sentence_lens_dev = [len(j) for j in sentences_words_dev]
    elif maintype == 'loadmodel_sentencerepresent':
        accIdx = str(168)
        decisionValency = 2
        nowPath = os.getcwd()

        PATH_chd = os.path.join(nowPath, 'wsj_idx168/my_model_chd_' + accIdx + '.h5')
        PATH_dec = os.path.join(nowPath, 'wsj_idx168/my_model_dec_' + accIdx + '.h5')
        model_chd = torch.load(PATH_chd)
        model_dec = torch.load(PATH_dec)

        # train_chd = os.path.join(nowPath, 'temp/predicted_train_chd' + accIdx + '.txt')
        # train_dec = os.path.join(nowPath, 'temp/predicted_train_dec' + accIdx + '.txt')

        abstractSts2File = "data/forWord2Vec/wsj-inf_2-21_dep_filter_10_abstractSts"  # should be corrected !!
        sentences_train = open(os.path.join(nowPath, abstractSts2File)).read().rstrip("\n").split("\n")
        sentences_words_train = [[int(j) for j in i.rstrip("\t").split('\t')] for i in sentences_train]
        sentence_lens_train = [len(j) for j in sentences_words_train]

        model_chd.eval()
        model_dec.eval()

        predictSentence(NN_child=model_chd, NN_decision=model_dec,
                          # predicted_chd=train_chd,
                          # predicted_dec=train_dec,
                          sentences_words=sentences_words_train,
                          sentence_lens=sentence_lens_train,
                          sentences_posSeq=sentences_words_train,
                          valency_size=decisionValency)
    elif main_type == 'limitto10':
        mainPath = '/home/hanwj/discrimitive_dmv_1/data'
        # langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        instc = 'wsj-inf_23_dep_all'
        # intdep = 'train_ndmv_init_conll'
        outstc = 'wsj-inf_23_dep'
        for i in range(1):
            in_stc = os.path.join(mainPath, instc)
            # in_dep = os.path.join(os.path.join(mainPath, lang), intdep)
            out_stc = os.path.join(mainPath, outstc)
            stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            print('all  num:  ' + str(len(stcs)))
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs if len(stc.rstrip("\n").rsplit("\n")) < 11]
            stcs_num = len(token_stcs)
            print('10  num:  ' + str(stcs_num))
            # stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            # token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])):  # one sentences
                    line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                           token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                           '\t' + token_stcs[i][j][6] + '\n'
                    wf.write(line)
                wf.write('\n')
            wf.close()
    elif main_type == 'conllu2oneline':
        mainPath = '/home/hanwj/Code/pennconverter/wsj_root' #'/home/hanwj/discrimitive_dmv_1/data'
        # langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        instc = 'WSJ_s23_tree_dep' #'wsj-inf_2-21_dep_filter_10'
        # intdep = 'train_ndmv_init_conll'
        outstc = 'WSJ_s23_tree_tags'  #'train_len10_lines'
        line_idx = 3
        max_len = 100000
        for i in range(1):
            in_stc = os.path.join(mainPath, instc)
            # in_dep = os.path.join(os.path.join(mainPath, lang), intdep)
            out_stc = os.path.join(mainPath, outstc)
            # stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            stcs = open(in_stc).read().rstrip('\n').split('\n\n')
            print('all  num:  ' + str(len(stcs)))
            token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs if len(stc.rstrip("\n").rsplit("\n")) < max_len]
            stcs_num = len(token_stcs)
            print('10  num:  ' + str(stcs_num))
            # stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            # token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                for j in range(len(token_stcs[i])-1):  # one sentences
                    # line = token_stcs[i][j][0] + '\t' + token_stcs[i][j][1] + '\t' + token_stcs[i][j][2] + '\t' + token_stcs[i][j][3] + '\t' + \
                    #        token_stcs[i][j][4] + '\t' + token_stcs[i][j][5] + \
                    #        '\t' + token_stcs[i][j][6] + '\n'
                    token = token_stcs[i][j][line_idx] + '\t'
                    wf.write(token)
                wf.write(token_stcs[i][len(token_stcs[i])-1][line_idx])
                wf.write('\n')
            wf.close()
    elif main_type == 'nums20':
        def is_not_num(temp):
            nums = [str(i) for i in range(10)]
            for i in nums:
                if i in temp:
                    return False
            return True
        mainPath = '/home/hanwj/Code/pennconverter/wsj_root'
        # langs = ['arabic', 'basque', 'czech', 'danish', 'dutch', 'portuguese', 'slovene', 'swedish']
        instc = 'WSJ_s23_tree_words'
        # intdep = 'train_ndmv_init_conll'
        outstc = 'WSJ_s23_tree_dep_words_0'
        for i in range(1):
            in_stc = os.path.join(mainPath, instc)
            # in_dep = os.path.join(os.path.join(mainPath, lang), intdep)
            out_stc = os.path.join(mainPath, outstc)
            # stcs = codecs.open(in_stc, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            stcs = open(in_stc).read().rstrip('\n').split('\n')
            print('all  num:  ' + str(len(stcs)))
            token_stcs = [[i if is_not_num(i) else '0' for i in stc.rstrip('\t').split('\t')] for stc in stcs]
            # token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\r\n").rsplit("\r\n")] for
            #               stc in stcs if len(stc.rstrip("\r\n").rsplit("\r\n")) < 100000]
            stcs_num = len(token_stcs)
            print('10  num:  ' + str(stcs_num))
            # stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
            # token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

            wf = codecs.open(out_stc, 'w', encoding='utf8')
            for i in range(stcs_num):
                # for j in range(len(token_stcs[i]) - 1):  # one sentences
                #     token = token_stcs[i][j][1] + '\t'
                #     wf.write(token)
                # wf.write(token_stcs[i][len(token_stcs[i]) - 1][1])
                wf.write(' '.join(token_stcs[i]))
                wf.write('\n')
            # wf.write('\n')
            wf.close()
    elif main_type == 'delet_root':
        mainPath = '/home/hanwj/Code/pennconverter/wsj_root'
        in_stc = os.path.join(mainPath, 'WSJ_s23_tree')# WSJ_s2-21_tree WSJ_s22_tree
        out_stc = os.path.join(mainPath, 'WSJ_s23_tree' + '_s')
        wf = codecs.open(out_stc, 'w', encoding='utf8')
        stcs = open(in_stc).read().rstrip('\n').split('\n')
        print('all  num:  ' + str(len(stcs)))
        # token_stcs = [[i if is_not_num(i) else '0' for i in stc.rstrip('\t').split('\t')] for stc in stcs]
        # token_stcs = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\r\n").rsplit("\r\n")] for
        #               stc in stcs if len(stc.rstrip("\r\n").rsplit("\r\n")) < 100000]
        stcs_num = len(stcs)
        # print('10  num:  ' + str(stcs_num))
        # stcs_dep = codecs.open(in_dep, 'r', encoding='utf8').read().rstrip('\n').split('\n\n')
        # token_stcs_dep = [[[k for k in line.rstrip('\t').split('\t')] for line in stc.rstrip("\n").rsplit("\n")] for stc in stcs_dep]

        wf = codecs.open(out_stc, 'w', encoding='utf8')
        for i in range(stcs_num):
            # for j in range(len(token_stcs[i]) - 1):  # one sentences
            #     token = token_stcs[i][j][1] + '\t'
            #     wf.write(token)
            # wf.write(token_stcs[i][len(token_stcs[i]) - 1][1])
            wf.write('('+stcs[i][6:])
            wf.write('\n')
        # wf.write('\n')
        wf.close()

if __name__ == "__main__":
    main_type = 'nums20' #'limitto10' #'loadmodel_sentencerepresent' #'conllforHDPDEP' #'pascalprocess'  # pascalinit_utf  #  loadmodel_sentencerepresent  # 'pascalprocess'
    main(main_type)