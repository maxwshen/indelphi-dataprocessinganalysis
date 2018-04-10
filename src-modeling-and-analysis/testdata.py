import _data

lib1_vo_spacers = ['overbeek_spacer_1', 'overbeek_spacer_3', 'overbeek_spacer_4', 'overbeek_spacer_5', 'overbeek_spacer_6', 'overbeek_spacer_7', 'overbeek_spacer_8', 'overbeek_spacer_9', 'overbeek_spacer_10', 'overbeek_spacer_11', 'overbeek_spacer_12', 'overbeek_spacer_13', 'overbeek_spacer_14', 'overbeek_spacer_15', 'overbeek_spacer_16', 'overbeek_spacer_17', 'overbeek_spacer_18', 'overbeek_spacer_19', 'overbeek_spacer_20', 'overbeek_spacer_21', 'overbeek_spacer_22', 'overbeek_spacer_23', 'overbeek_spacer_24', 'overbeek_spacer_25', 'overbeek_spacer_26', 'overbeek_spacer_27', 'overbeek_spacer_28', 'overbeek_spacer_29', 'overbeek_spacer_30', 'overbeek_spacer_32', 'overbeek_spacer_33', 'overbeek_spacer_34', 'overbeek_spacer_35', 'overbeek_spacer_36', 'overbeek_spacer_37', 'overbeek_spacer_38', 'overbeek_spacer_39', 'overbeek_spacer_40', 'overbeek_spacer_41', 'overbeek_spacer_43', 'overbeek_spacer_44', 'overbeek_spacer_45', 'overbeek_spacer_46', 'overbeek_spacer_47', 'overbeek_spacer_48', 'overbeek_spacer_49', 'overbeek_spacer_50', 'overbeek_spacer_51', 'overbeek_spacer_52', 'overbeek_spacer_53', 'overbeek_spacer_54', 'overbeek_spacer_55', 'overbeek_spacer_56', 'overbeek_spacer_58', 'overbeek_spacer_59', 'overbeek_spacer_60', 'overbeek_spacer_61', 'overbeek_spacer_62', 'overbeek_spacer_63', 'overbeek_spacer_64', 'overbeek_spacer_65', 'overbeek_spacer_66', 'overbeek_spacer_68', 'overbeek_spacer_69', 'overbeek_spacer_70', 'overbeek_spacer_71', 'overbeek_spacer_72', 'overbeek_spacer_73', 'overbeek_spacer_74', 'overbeek_spacer_75', 'overbeek_spacer_76', 'overbeek_spacer_77', 'overbeek_spacer_78', 'overbeek_spacer_79', 'overbeek_spacer_80', 'overbeek_spacer_81', 'overbeek_spacer_82', 'overbeek_spacer_83', 'overbeek_spacer_84', 'overbeek_spacer_85', 'overbeek_spacer_86', 'overbeek_spacer_87', 'overbeek_spacer_88', 'overbeek_spacer_89', 'overbeek_spacer_90', 'overbeek_spacer_91', 'overbeek_spacer_92', 'overbeek_spacer_93', 'overbeek_spacer_94', 'overbeek_spacer_95']
lib1_1873 = [str(s) for s in range(1873)]

# dataset = _data.load_dataset('1027-mESC-Lib1-Cas9-Tol2-Biorep1-r1', exp_subset = lib1_vo_spacers, exp_subset_col = 'Designed Name')
# dataset = _data.load_dataset('1027-mESC-Lib1-Cas9-Tol2-Biorep1-r1-controladj', exp_subset = lib1_vo_spacers, exp_subset_col = 'Designed Name')

# d1 = _data.load_dataset('VO-spacers-HEK293-48h-r1')
# d2 = _data.load_dataset('VO-spacers-HEK293-48h-r2')
# d3 = _data.load_dataset('VO-spacers-HEK293-48h-r3')


d = _data.load_l3_dataset('VO-spacers-HCT116-4h-controladj')
# d = _data.load_l3_dataset('Lib1-mES-controladj')

import code; code.interact(local=dict(globals(), **locals()))
# dataset = _data.load_dataset('1215-ERev2-Dislib')
