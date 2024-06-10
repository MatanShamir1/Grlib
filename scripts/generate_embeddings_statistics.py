import dill
import torch

def main():
    task_num = 0
    for goal in ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']:
			for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
                with open(f'embeddings_dict{task_num}.pkl', 'rb') as emb_file:
                    embeddings_dict = dill.load(emb_file)
                    goal_embedding = embeddings_dict['actual_goal']
                    embeddings_dict.pop('actual_goal')
                    print(embeddings_dict)
                    for (goal, embedding) in embeddings_dict.items():
                        curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-goal_embedding)))
                        print (f'goal: {goal}, similarity: {curr_similarity}')

if __name__ == "__main__":
	main()