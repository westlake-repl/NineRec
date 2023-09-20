import pandas as pd
import re
'''
convert the user-item pairs into item sequences format.
'''
df_1 = pd.read_csv('Bili_2M_pair.csv', iterator = True, header=None)
loop = True
chunkSize = 10000
chunks = []
while loop:
    try:
        chunk = df_1.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")
df_1 = pd.concat(chunks, ignore_index = True)

df_1.columns = ['video_id', 'user_id', 'timestamp']
print(df_1[:5])

df_1 = df_1.sort_values('timestamp')

f = open('Bili_2M_behaviour.tsv', 'w+')
for user, hist in df_1.groupby(['user_id']):
    videos = list(hist['video_id'])
    f.write(str(user)+'\t')
    f.write(str(videos[0]))
    for video in videos[1:]:
        f.write(' '+str(video))
    f.write('\n')

f.close()


