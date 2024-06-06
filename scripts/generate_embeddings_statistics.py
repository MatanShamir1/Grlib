import dill
import torch

def main():
    with open("embeddings_dict.pkl", 'rb') as emb_file:
        embeddings_dict = dill.load(emb_file)
        goal_embedding = embeddings_dict['actual_goal']
        embeddings_dict.pop('actual_goal')
        print(embeddings_dict)
        for (goal, embedding) in embeddings_dict.items():
            curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-goal_embedding)))
            print (f'goal: {goal}, similarity: {curr_similarity}')

if __name__ == "__main__":
	main()