



import pickle
import os
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed, parallel_backend

from sklearn.metrics import pairwise_distances
import data_core.morphology as morphology



def size_apply_tallaje(size, tallaje, category):
    '''
    Obtain the size of the item taking into account the 'tallaje' of the item
    '''
    size = morphology.convert_size(size, category)

    if tallaje in ['grande', 'grande_1']:
        size = morphology.get_next_size(size, -1)
    elif tallaje == 'grande_2':
        size = morphology.get_next_size(size, -2)
    elif tallaje in ['pequeno', 'pequeno_1']:
        size = morphology.get_next_size(size, 1)
    elif tallaje == 'pequeno_2':
        size = morphology.get_next_size(size, 2)
    else:
        size = size
    return size


def get_product_from_repo(repo):
    list_product_id = []
    list_group = []
    list_group_size = []
    list_size = []
    list_family = []
    list_size_reference = []
    list_tallaje = []
    list_fit = []
    list_gomaCintura = []
    list_largoCM = []
    list_category = []
    list_contornoCinturaCm = []
    list_contornoCaderaCm = []
    list_mangaLargoCm = []
    list_contornoPechoCm = []

    for pr_id in repo.products:
        # print(pr_id)
        list_product_id.append(pr_id)
        list_group.append(repo.products[pr_id].group)
        list_group_size.append(repo.products[pr_id].group + '_' + repo.products[pr_id].size
                               if (repo.products[pr_id].group is not None and
                                   repo.products[pr_id].size is not None) else np.nan)

        list_size.append(repo.products[pr_id].size)
        list_family.append(repo.products[pr_id].family)
        list_size_reference.append(repo.products[pr_id].size_reference)
        list_category.append(repo.products[pr_id].get_category())

        list_tallaje.append(repo.products[pr_id].characteristics['tallaje']
                            if (repo.products[pr_id].characteristics is not None and 'tallaje' in
                                repo.products[pr_id].characteristics.keys()) else np.nan)

        list_fit.append(repo.products[pr_id].characteristics['fit']
                        if (repo.products[pr_id].characteristics is not None and 'fit' in
                            repo.products[pr_id].characteristics.keys()) else np.nan)

        list_gomaCintura.append(repo.products[pr_id].characteristics['gomaCintura']
                                if (repo.products[pr_id].characteristics is not None and 'gomaCintura'
                                    in repo.products[pr_id].characteristics.keys()) else np.nan)

        list_largoCM.append(repo.products[pr_id].characteristics['largoCM']
                            if (repo.products[pr_id].characteristics is not None and 'largoCM' in
                                repo.products[pr_id].characteristics.keys()) else np.nan)

        list_contornoCinturaCm.append(repo.products[pr_id].characteristics['contornoCinturaCm']
                                      if (repo.products[pr_id].characteristics is not None and 'contornoCinturaCm'
                                          in repo.products[pr_id].characteristics.keys()) else np.nan)

        list_contornoCaderaCm.append(repo.products[pr_id].characteristics['contornoCaderaCm']
                                     if (repo.products[pr_id].characteristics is not None and 'contornoCaderaCm'
                                         in repo.products[pr_id].characteristics.keys()) else np.nan)
        list_mangaLargoCm.append(repo.products[pr_id].characteristics['mangaLargoCm']
                                 if (repo.products[pr_id].characteristics is not None and 'mangaLargoCm'
                                     in repo.products[pr_id].characteristics.keys()) else np.nan)
        list_contornoPechoCm.append(repo.products[pr_id].characteristics['contornoPechoCm']
                                    if (repo.products[pr_id].characteristics is not None and 'contornoPechoCm'
                                        in repo.products[pr_id].characteristics.keys()) else np.nan)


    # tallaje
    list_size_tallaje = list(map(size_apply_tallaje, list_size, list_tallaje, list_category))

    df_product = pd.DataFrame({#'product_id': list_product_id,
                               'group': list_group_size, #list_group,
                               'size': list_size,
                               'size_tallaje': list_size_tallaje,
                               'family': list_family,
                               'size_reference': list_size_reference,
                               'tallaje': list_tallaje,
                               'fit': list_fit,
                               'gomaCintura': list_gomaCintura,
                               'largoCM': list_largoCM,
                               'product_category': list_category,
                               'contornoCinturaCm': list_contornoCinturaCm,
                               'contornoCaderaCm': list_contornoCaderaCm,
                               'mangaLargoCm': list_mangaLargoCm,
                               'contornoPechoCm': list_contornoPechoCm})

    # change the nan of the gomaCintura to -1
    df_product['gomaCintura'] = df_product['gomaCintura'].fillna(-1)
    df_product = df_product.dropna(subset=['product_category'])

    # df_product['product_category'].value_counts(dropna=False)
    # df_product.describe()

    column_sort = ['size',  # 0
                   'family',  # 1
                   'size_tallaje',  # 2 'size_tallaje' era 'size_reference'
                   'tallaje',  # 3
                   'fit',  # 4
                   'gomaCintura',  # 5
                   'largoCM',  # 6
                   'contornoCinturaCm',  # 7
                   'contornoCaderaCm',  # 8
                   'mangaLargoCm',  # 9
                   'contornoPechoCm',  # 10
                   'group',  # 11
                   'product_category']  # 12

    df = df_product.reindex(column_sort, axis=1)

    df_product_top = df[df['product_category'] == 'top'].drop(['product_category'], axis=1)
    df_product_trousers = df[df['product_category'] == 'trousers'].drop(['product_category'], axis=1)
    df_product_dress = df[df['product_category'] == 'dress'].drop(['product_category'], axis=1)
    df_product_outer = df[df['product_category'] == 'outer'].drop(['product_category'], axis=1)
    df_product_skirt = df[df['product_category'] == 'skirt'].drop(['product_category'], axis=1)
    df_product_accessory = df[df['product_category'] == 'accessory'].drop(['product_category'], axis=1)
    df_product_jumpsuit = df[df['product_category'] == 'jumpsuit'].drop(['product_category'], axis=1)

    dict_category_product = {'top': df_product_top,
                             'trousers': df_product_trousers,
                             'dress': df_product_dress,
                             'outer': df_product_outer,
                             'skirt': df_product_skirt,
                             'accessory': df_product_accessory,
                             'jumpsuit': df_product_jumpsuit}

    #######################################################################################################################
    # save

    # df_product_top.to_csv('/home/darya/Documents/projects/temp_plots/df_product_top.csv')
    # df_product_trousers.to_csv('/home/darya/Documents/projects/temp_plots/df_product_trousers.csv')
    # df_product_dress.to_csv('/home/darya/Documents/projects/temp_plots/df_product_dress.csv')
    # df_product_outer.to_csv('/home/darya/Documents/projects/temp_plots/df_product_outer.csv')
    # df_product_skirt.to_csv('/home/darya/Documents/projects/temp_plots/df_product_skirt.csv')
    # df_product_accessory.to_csv('/home/darya/Documents/projects/temp_plots/df_product_accessory.csv')
    # df_product_jumpsuit.to_csv('/home/darya/Documents/projects/temp_plots/df_product_jumpsuit.csv')

    return dict_category_product


def get_distance_product_numeric(row1, row2, category):
    '''Calculate the distance between two products.
    Cod adapted from the size_model->statistical_model-> get_distance(prod1, prod2)'''
    max_distance = 1e6

    category_chars_to_be_measure = {
        'trousers': {
            7: True, # 'contornoCinturaCm': True,
            8: True # 'contornoCaderaCm': True
        },
        'skirt': {
            7: True, # 'contornoCinturaCm'
            8: True # 'contornoCaderaCm'
        },
        'top': {
            9: True, # 'mangaLargoCm'
            10: True, # 'contornoPechoCm'
            7: True, # 'contornoCinturaCm'
        },
        'dress': {
            10: True,  # 'contornoPechoCm'
            7: True, # 'contornoCinturaCm'
            8: True, # 'contornoCaderaCm'
            9: True, # 'mangaLargoCm'
        },
        'jumpsuit': {
            10: True,  # 'contornoPechoCm'
            7: True, # 'contornoCinturaCm'
            8: True, # 'contornoCaderaCm'
            9: True, # 'mangaLargoCm'
        },
        'outer': {
            10: True,  # 'contornoPechoCm'
            7: True, # 'contornoCinturaCm'
            8: True, # 'contornoCaderaCm'
            9: True, # 'mangaLargoCm'
        }
    }

    try:
        distance = 0.0
        distance += np.abs(row1[6] - row2[6])
        for char in category_chars_to_be_measure[category].keys():
            distance += np.abs(row1[char] - row2[char])
        return distance

    except:
        max_distance


def get_distance_product_filter_numeric(df, category, row_index):
    X = df.to_numpy()
    product_group = df['group'][row_index]
    row = X[row_index]
    # print(row_index)
    data = np.delete(X, row_index, axis=0)

    data = np.delete(data, np.argwhere(data[:, 0] != row[0]), axis=0)  # size
    data = np.delete(data, np.argwhere(data[:, 1] != row[1]), axis=0)  # family

    data = np.delete(data, np.argwhere(data[:, 4] != row[4]), axis=0)  # fit

    data = np.delete(data, np.argwhere(data[:, 5] != row[5]), axis=0)  # gomaCintura

    if row[0] == 'UNQ':
        data = np.delete(data, np.argwhere(data[:, 0] != 'UNQ'), axis=0)  # UNQ size
        data = np.delete(data, np.argwhere(data[:, 3] != row[3]), axis=0)  # tallaje

    if data.shape[0] > 0:

        data_numeric = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        row_numeric = pd.to_numeric(pd.Series(row), errors='coerce').fillna(0).to_numpy()

        pw_dist = pairwise_distances(data_numeric,
                                     row_numeric.reshape(1, -1),
                                     metric=get_distance_product_numeric,
                                     category=category,
                                     n_jobs=-1)

        dict_product_distance = dict(zip(data[:, -1], pw_dist[:, 0]))
        if product_group in dict_product_distance: del dict_product_distance[product_group]

        if len(dict_product_distance) > 0:
            return dict_product_distance


def calculate_product_distance(df, category):

    result = {}
    for row_index in range(0, df.shape[0]):
        product_group = df['group'][row_index]
        dict_product_distance = get_distance_product_filter_numeric(df, category, row_index)
        result[product_group] = dict_product_distance
    return result


def calculate_product_distance_parallel(df, category, n_jobs=6):
    with parallel_backend('threading', n_jobs=n_jobs):
        dist = Parallel()(delayed(get_distance_product_filter_numeric)(df, category, row_index)
                                for row_index in range(0, df.shape[0]))
    dict_product_distance = dict(zip(df['group'], dist))
    # remove None items
    dict_product_distance = {k: v for k, v in dict_product_distance.items() if v is not None}
    return dict_product_distance


repo_file = '/var/lib/lookiero/repo_v2.5.0.obj'
repo = pickle.load(open(repo_file, 'rb'))

dict_category_product = get_product_from_repo(repo)

for category in dict_category_product.keys():

    df = dict_category_product[category].reset_index(drop=True)

    # to test
    # df = df.iloc[0:1000]

    print('Calculating distance of products of category-->', category)
    t_start = time.time()
    result_jobs = calculate_product_distance_parallel(df, category, n_jobs=6)
    print("Time find items: {:.3f}s".format(time.time() - t_start))

    #  save
    path_save = ('/home/darya/Documents/projects/temp_plots')
    file_name_save = 'product_distance_' + str(category) + '.pkl'
    f = open(os.path.join(path_save, file_name_save), 'wb')
    pickle.dump(result_jobs, f)
    f.close()
    print('Saving distance of the products to: ', os.path.join(path_save, file_name_save))

# test open

f = open(os.path.join(path_save, file_name_save), 'rb')
unpickler = pickle.Unpickler(f)
tes_open = unpickler.load()






# for row_index in range(0, X.shape[0]):
#
#     product_group = df['group'][row_index]
#     row = X[row_index]
#     # print(row_index)
#     data = np.delete(X, row_index, axis=0)
#
#     data = np.delete(data, np.argwhere(data[:, 0] == 'UNQ'), axis=0)  # UNQ size
#     data = np.delete(data, np.argwhere(data[:, 1] != row[1]), axis=0)  # family
#     data = np.delete(data, np.argwhere(data[:, 2] != row[2]), axis=0)  # size_reference
#     data = np.delete(data, np.argwhere(data[:, 3] != row[3]), axis=0)  # tallaje
#     data = np.delete(data, np.argwhere(data[:, 4] != row[4]), axis=0)  # fit
#     data = np.delete(data, np.argwhere(data[:, 5] != row[5]), axis=0)  # gomaCintura
#
#     if data.shape[0] > 0:
#
#         data_numeric = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
#         row_numeric = pd.to_numeric(pd.Series(row), errors='coerce').fillna(0).to_numpy()
#
#         # t_start = time.time()
#         pw_dist = pairwise_distances(data_numeric,
#                                      row_numeric.reshape(1, -1),
#                                      metric=get_distance_product_numeric,
#                                      category=category,
#                                      n_jobs=-1)
#
#         dict_prodect_distance = dict(zip(data[:, -1], pw_dist[:, 0]))
#         # TODO: remove current product_group from the fnal distance dict
#         if product_group in dict_prodect_distance: del dict_prodect_distance[product_group]
#         # print("Time find items: {:.3f}s".format(time.time() - t_start))
#         result[product_group] = dict_prodect_distance