import pandas as pd
from collections import defaultdict

libraries = defaultdict(lambda: defaultdict(list))

curr_dir = '/cluster/mshen/prj/mmej_figures/data/'

df = pd.read_csv(curr_dir + 'overbeek_data.csv')
mmc2 = pd.read_csv(curr_dir + 'mmc2.txt')

# Read in mmc2, getting a mapping from genomic location to spacer name
chros, starts, ends, spacers = [], [], [], []
for idx, row in mmc2.iterrows():
  loc = row['hg19 location'].strip()
  chro = loc.split(':')[0]
  [start, end] = loc.split(':')[1].split('-')
  start, end = int(start), int(end)
  chros.append(chro)
  starts.append(start)
  ends.append(end)
  spacers.append(row['Spacer'])

def find_spacer(w):
  chro = w.split(':')[0]
  [start, end] = w.split(':')[1].split('-')
  start, end = int(start), int(end)

  best_dist = 500
  best_spacer = ''
  for c, s, e, spacer in zip(chros, starts, ends, spacers):
    if c == chro:
      if abs(start - s) <= 1:
        if abs(end - e) <= 1:
          dist = abs(start - s) + abs(end - e)
          if dist < best_dist:
            best_dist = dist
            best_spacer = spacer
  return best_spacer

# Iterate over overbeek_data to find srr_ids for spacers by experiment group / library
for idx, row in df.iterrows():
  srr_id = row['Run']
  srr_num = int(srr_id.replace('SRR', ''))
  name = row['Library_Name']
  ws = name.split('_')

  # Process standard experiment set 
  if srr_num <= 3701229:
    spacer = ''
    time = ''
    rep = ''
    celltype = ''
    for w in ws:
      if w[:3] == 'chr':
        spacer = find_spacer(w)
      if w in ['WT', '48hr', '24hr', '16hr', '8hr', '4hr']:
        time = w
        if time != 'WT':
          time = time[:-1] # change hr into h
      if w in ['HCT116', 'HEK293', 'K562']:
        celltype = w
      if w in ['R1', 'R2', 'R3']:
        rep = w.lower()

    if time == 'WT':
      lib_name = 'Lib-VO-spacers-%s-%s' % (celltype, time)
    else:
      lib_name = 'Lib-VO-spacers-%s-%s-%s' % (celltype, time, rep)
  
  # Process NU7441 nhej knockouts
  elif 3702337 <= srr_num <= 3702360:
    rep = ''
    treated = ''
    conc = ''
    for w in ws:
      if w[:3] == 'chr':
        spacer = find_spacer(w)
      if w in ['treated', 'untreated']:
        treated = w
      if w in ['1', '2', '3', '4', '5']:
        conc = w
      if w in ['R1', 'R2']:
        rep = w.lower()

      if treated == 'untreated':
        lib_name = 'Lib-VO-NU7441-untreated-%s' % (rep)
      else:
        lib_name = 'Lib-VO-NU7441-conc%s-%s' % (conc, rep)

  # Not an experiment to include in libraries
  else:
    continue

  assert spacer != '', 'No spacer found'
  # Place srr_ids and spacers into appropriate libraries
  spacer_string = 'overbeek_spacer_%s' % (spacer)
  assert spacer_string not in libraries[lib_name]['Name'], '%s already exists!' % (spacer_string)

  libraries[lib_name]['Local Name'].append(srr_id)
  libraries[lib_name]['Name'].append(spacer_string)
  libraries[lib_name]['Designed Name'].append(spacer_string)

# Save libraries
out_dir = curr_dir + 'Libraries/'
for lib_name in sorted(libraries):
  lib_d = libraries[lib_name]
  d = pd.DataFrame(lib_d)
  print 'Wrote library %s with shape %s' % (lib_name, d.shape)
  d = d[['Local Name', 'Name', 'Designed Name']]
  d = d.sort_values(by = 'Name')
  d.to_csv(out_dir + lib_name + '.csv', index = False)


