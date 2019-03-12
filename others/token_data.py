#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:37:29 2019

@author: fengjiaxin
"""

from __future__ import print_function

import os
import progressbar
import Smipar


def convertList2Str(index_list):
    ret_str = ''
    for index in index_list:
        ret_str += str(index) + ' '
    return ret_str[:-1]


def token_main():
    data_directory = '../data'
    processed_data_filename = 'reactionSmiles_processed_cano_data.rsmi'

    vocab_dir = 'vocab'
    vocab_reactants_filename = 'vocab_reactants.txt'
    vocab_products_filename = 'vocab_products.txt'

    training_reactants_filename = 'reactionSmiles_training_ids.reactants'
    training_products_filename = 'reactionSmiles_training_ids.products'

    train_reactants_filename = 'reactionSmiles_train_ids.reactants'
    train_products_filename = 'reactionSmiles_train_ids.products'

    dev_reactants_filename = 'reactionSmiles_dev_ids.reactants'
    dev_products_filename = 'reactionSmiles_dev_ids.products'

    processed_data_filepath = os.path.join(data_directory, processed_data_filename)
    vocab_reactants_path = os.path.join(data_directory, vocab_dir, vocab_reactants_filename)
    vocab_products_path = os.path.join(data_directory, vocab_dir, vocab_products_filename)

    training_reactants_path = os.path.join(data_directory, training_reactants_filename)
    training_products_path = os.path.join(data_directory, training_products_filename)

    train_reactants_path = os.path.join(data_directory, train_reactants_filename)
    train_products_path = os.path.join(data_directory, train_products_filename)

    dev_reactants_path = os.path.join(data_directory, dev_reactants_filename)
    dev_products_path = os.path.join(data_directory, dev_products_filename)

    data_length = 1925386
    bar = progressbar.ProgressBar(maxval=data_length)

    # reactionSmiles list
    rsmi_data = []

    rsmi_data = []
    with open(processed_data_filepath, 'r') as data_file:
        for line in data_file:
            line = line.strip('\n')
            rsmi_data.append(line)

    vocab_reactants_list = []
    vocab_products_list = []

    # load vocab
    with open(vocab_reactants_path, 'r') as rp:
        for line in rp:
            react_vocab = line.strip('\n')
            vocab_reactants_list.append(react_vocab)

    with open(vocab_products_path, 'r') as pp:
        for line in pp:
            prod_vocab = line.strip('\n')
            vocab_products_list.append(prod_vocab)

    token_len = []

    error_rsmi = {}

    '''
    原始数据的格式为reactants>regants>products
    需要转换成regants>products>reactants
    其中dev测试集格式为regants>products
    所以转换成化学翻译语言的格式为
    source_data :regants>products
    target_data :reactants 

    !! 但是在去掉原子映射并未进行处理

    '''
    token_len = []

    error_rsmi = {}

    bar.start()
    with open(training_reactants_path, 'w') as training_reactants:
        with open(training_products_path, 'w') as training_products:
            for i, rsmi in enumerate(rsmi_data):

                reactant_list = []
                agent_list = []
                product_list = []

                try:
                    split_rsmi = rsmi.split('>')
                    reactants = split_rsmi[0].split('.')
                    agents = split_rsmi[1].split('.')
                    products = split_rsmi[2].split('.')

                    for reactant in reactants:
                        reactant_list += Smipar.parser_list(reactant)
                        reactant_list += '.'
                    for agent in agents:
                        agent_list += Smipar.parser_list(agent)
                        agent_list += '.'
                    for product in products:
                        product_list += Smipar.parser_list(product)
                        product_list += '.'

                    reactant_list.pop()  # to pop last '.'
                    agent_list.pop()
                    product_list.pop()

                    agent_list += '>'
                    agent_list += product_list

                    # 此时agent_list是原始数据的regants 和 products;reactant_list是reactants

                    token_len.append((len(agent_list), len(reactant_list)))

                    #reactants_ids_list = [vocab_reactants_list.index(r) for r in reactant_list]
                    reactants_ids_str = convertList2Str(reactant_list)

                    #products_ids_list = [vocab_products_list.index(r) for r in agent_list]
                    products_ids_str = convertList2Str(agent_list)

                    training_reactants.write(reactants_ids_str + '\n')
                    training_products.write(products_ids_str + '\n')

                except:
                    error_rsmi.update({i: rsmi})

                bar.update(i)

    bar.finish()

    #######################################

    count = 0
    with open(training_reactants_path, 'r') as data_file:
        with open(train_reactants_path, 'w') as train_file:
            with open(dev_reactants_path, 'w') as dev_file:
                for i, line in enumerate(data_file):
                    reactants_ids = line.strip('\n')
                    if i % 100:
                        train_file.write(reactants_ids + '\n')
                    else:
                        dev_file.write(reactants_ids + '\n')
                        count += 1
                    bar.update(i)
    bar.finish()

    count = 0
    with open(training_products_path, 'r') as data_file:
        with open(train_products_path, 'w') as train_file:
            with open(dev_products_path, 'w') as dev_file:
                for i, line in enumerate(data_file):
                    products_ids = line.strip('\n')
                    if i % 100:
                        train_file.write(products_ids + '\n')
                    else:
                        dev_file.write(products_ids + '\n')
                        count += 1
                    bar.update(i)
    bar.finish()


if __name__ == '__main__':
    token_main()



