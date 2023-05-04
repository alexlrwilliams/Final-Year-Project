import os
import torch.utils.data

if __name__ == '__main__':
    test = os.listdir("../data/videos/utterance")
    new = [x.split('.')[0] for x in test]
    ahh = list(torch.load('../data/hubert-embeddings.pt').keys())
    for pic in new:
        x = pic.replace("_u", "")
        if not x in ahh:
            print(pic)
    # dataframe_dir = os.path.join("..", CONFIG.DATA_PATH)
    # df = pd.read_csv(dataframe_dir, encoding = "ISO-8859-1")
    # for index, data in df.iterrows():
    #     id = data['SCENE']
    #     print(os.path.join("..","data","frames","utterances_final",id+"_u"))
    #     print(os.path.exists(os.path.join("..","data","frames","utterances_final",id+"_u")))
    #     break