TARGET = 'target'

FEATURES = [
    'days_g2',
    'gw_g10',
    'gw_g8',
    'GOC_to_g9',
    'succ_dep_g10',
    'succ_dep_cnt_g9',
    'ini_bon_g10',
    'turnover_last_20days',
    'to_l5_l20',
    'SE_GI_total_70days',
    'days_since_last_SE_GI',
    'SE_GI_max_datediff'
]

MODEL_PARAMS = {
    'depth': 8,
    'n_estimators': 215,
    'l2_leaf_reg': 1,
    'border_count': 32,
    'random_strength': 0.765,
    'eta': 0.45989
}
