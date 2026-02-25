import torch
import pandas as pd
import numpy as np
import torch.nn as nn
#from tensorflow.python.ops.gradient_checker import compute_gradient
import torch
import numpy as np
from samplers import Langevin_TSP, HiSS_TSP, GWG_TSP, PT_DMALA_TSP, ACS_TSP, ELangevin_TSP
import argparse
import torch.nn as nn
import time
import torch




def jaccard_similarity(path1, path2):
    """Returns the Jaccard similarity of directed edges between two tours."""
    cycle1 = path1 + [path1[0]]
    cycle2 = path2 + [path2[0]]
    edges1 = set(zip(cycle1[:-1], cycle1[1:]))
    edges2 = set(zip(cycle2[:-1], cycle2[1:]))
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)
    return len(intersection) / len(union) if union else 0.0

# Example usage:


def pairwise_mismatch_count(path1, path2):
    path1=path1+[path1[0]]
    path2 = path2 + [path2[0]]

    distance = 0
    for i in range(len(path1)):
        for j in range(i + 1, len(path1)):
            # Check if the pairwise order is different in the two paths
            if (path1[i], path1[j]) != (path2[i], path2[j]):
                distance += 1
    return distance

def cost(one_hot_permutations,city_coords):
    """
    Compute the average tour cost and standard deviation for a batch of one-hot encoded permutations.

    Args:
        one_hot_permutations (torch.Tensor): Tensor of shape [batch_size, num_cities, num_cities]
                                             representing one-hot encoded tours.

    Returns:
        avg_cost (float): Average tour cost over the batch.
        std_cost (float): Standard deviation of the tour costs.
    """
    batch_size = one_hot_permutations.shape[0]
    total_costs = []

    for idx in range(batch_size):
        one_hot_permutation = one_hot_permutations[idx]

        # Convert one-hot encoding to tour coordinates
        permuted_cities = torch.matmul(one_hot_permutation.float(), city_coords)  # Shape: [num_cities, 2]

        # Compute tour cost (Euclidean distance)
        total_cost = 0.0
        for i in range(permuted_cities.shape[0] - 1):
            total_cost += euclidean_distance(permuted_cities[i], permuted_cities[i + 1])

        # Return to starting city
        total_cost += euclidean_distance(permuted_cities[-1], permuted_cities[0])

        total_costs.append(total_cost)

    # Convert to tensor for statistical computation
    total_costs_tensor = torch.tensor(total_costs, device=one_hot_permutations.device)
    max_cost = total_costs_tensor.max().item()
    min_cost = total_costs_tensor.min().item()
    median_cost = total_costs_tensor.median().item()
    range_cost = max_cost - min_cost
    print("Range: "+str(range_cost))
    avg_cost = total_costs_tensor.mean().item()
    std_cost = total_costs_tensor.std().item()

    return avg_cost, std_cost, max_cost, min_cost, median_cost, range_cost

def to_one_hot_batch(chain, num_cities):
    """
    Convert a list of city tours into a batch of one-hot encoded tensors.

    Args:
        chain (list): List of tours, where each tour is a list of city indices.
        num_cities (int): Total number of cities.

    Returns:
        one_hot_batch (torch.Tensor): Tensor of shape [batch_size, num_cities, num_cities].
    """
    batch_size = len(chain)
    indices = torch.tensor(chain, dtype=torch.long)  # Shape: [batch_size, num_cities]
    one_hot_batch = torch.zeros(batch_size, num_cities, num_cities,device=indices.device)
    one_hot_batch.scatter_(2, indices.unsqueeze(2), 1.0)  # Create one-hot encoding
    return one_hot_batch


# Load the city coordinates into a DataFrame
def load_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    data_started = False
    data = []
    for line in lines:
        if line.strip() == "NODE_COORD_SECTION":
            data_started = True
            continue
        if data_started:
            if line.strip() == "EOF":
                break
            parts = line.strip().split()
            city_num, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            data.append([city_num, x, y])

    df = pd.DataFrame(data, columns=["City", "X", "Y"])
    return df



# Euclidean distance for TSP
def euclidean_distance(city1, city2):
    return torch.norm(city1 - city2, p=2)





class TSP(nn.Module):
    def __init__(self, data_dim, num_cls, city_coords):
        super().__init__()
        self.data_dim = data_dim
        self.num_cls = num_cls
        self.city_coords = city_coords
        self.num_cities = data_dim

    def forward(self, one_hot_permutation, r=5,c=2, penalty=False):
        """Compute the total tour distance for a given one-hot encoded permutation."""
        total_cost = 0.0

        # Convert one-hot encoding to soft indices using matrix multiplication
        # This keeps the operation differentiable
        #permutation_indices = torch.matmul(one_hot_permutation, torch.arange(one_hot_permutation.shape[-1], device=one_hot_permutation.device)).long()

        # Gather the city coordinates in tour order

        permuted_cities = torch.matmul(one_hot_permutation.float(), self.city_coords)  # Shape: [batch_size, num_cities, 2]

        # Sum over distances between consecutive cities
        for i in range(permuted_cities.shape[1] - 1):
            total_cost += euclidean_distance(permuted_cities[:, i], permuted_cities[:, i + 1])

        # Return to the starting city
        total_cost += euclidean_distance(permuted_cities[:, -1], permuted_cities[:, 0])


        if penalty:
            row_penalty = torch.sum((one_hot_permutation.sum(dim=1) - 1) ** 2)  # Row sum should be 1
            col_penalty = torch.sum((one_hot_permutation.sum(dim=0) - 1) ** 2)
            uniqueness_penalty =r*row_penalty + c*col_penalty
        else:
            uniqueness_penalty=0

        return -total_cost + uniqueness_penalty  # Negative for minimization # Negative for minimization



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)



    k=args.cities
    cities_df = load_data('./eil40.tsp')
    # Convert to tensor for computation
    city_coords = torch.tensor(cities_df.iloc[:k][['X', 'Y']].values, dtype=torch.float32).to(device)

    SE=[]
    MAE = []
    Prob_states = []
    TIME = []
    # dimension=[(1,2),(2,3),(3,4),(4,5)]
    samp = args.sampler
    DATA_DIM = k ### Number of dimensions
    NUM_CLS = DATA_DIM ### Number of classes for each dimension
    N=1

    model = TSP(data_dim=DATA_DIM, num_cls=NUM_CLS, city_coords=city_coords)
    model.to(device)
    print(args.step_size, args.step_size_a, args.eta)
    print("START")
    y = torch.randperm(model.num_cities).unsqueeze(0)
    start=y[0].tolist()
    print(start)
    if samp == "dmala":
        sampler = Langevin_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L, temp=2., step_size=args.step_size,
                                              mh=True)
    elif samp == "dula":
        sampler = Langevin_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L, temp=2., step_size=args.step_size,
                                              mh=False)
    elif samp == "edmala":
        sampler = ELangevin_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L, temp=2., step_size=args.step_size, step_size_a=args.step_size_a, eta=args.eta,
                                              mh=True)
    elif samp == "edula":
        sampler = ELangevin_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L, temp=2., step_size=args.step_size, step_size_a=args.step_size_a, eta=args.eta,
                                              mh=False)
    elif samp == "gwg":
        sampler = GWG_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L, temp=2., step_size=args.step_size)
    elif samp == "hiss":
        sampler = HiSS_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G, temp=2., mh=True, step_size=args.step_size, eta=args.eta, score_sweeps=args.L)
    elif samp == "hiss-nomh":
        sampler = HiSS_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G, temp=2., mh=False, step_size=args.step_size, eta=args.eta, score_sweeps=args.L)

    elif samp=='pt+dmala':
        sampler= PT_DMALA_TSP(model.data_dim, num_cls=model.num_cls, n_steps=args.G*args.L,
                                               n_chains=args.tempchains, step_size=args.step_size, swap_interval=100,
                                               mh=True)
    elif args.sampler == 'acs':
        sampler = ACS_TSP(dim=model.data_dim,max_val=model.data_dim, num_cls=model.num_cls,
                                                        n_steps=args.G * args.L,
                                                        mh=True,
                                                        fixed_proposal=False, approx=True, multi_hop=False,
                                                        num_cycles=args.num_cycles,
                                                        num_iters=10,
                                                        mean_stepsize=args.step_size,
                                                        initial_balancing_constant=args.initial_balancing_constant,
                                                        burnin_adaptive=args.burnin_adaptive,
                                                        burnin_budget=args.burnin_budget,
                                                        burnin_lr=args.burnin_lr,
                                                        sbc=args.use_manual_EE,
                                                        big_step=5,
                                                        big_bal=args.big_bal,
                                                        small_step=0.05,
                                                        small_bal=args.small_bal,
                                                        min_lr=args.min_lr,
                                                        device=device)
    else:
        assert False, 'Not implemented'


    x = y
    x_a = torch.zeros(N, DATA_DIM, NUM_CLS)

    x = x.to(device)
    x_a = x_a.to(device)
    chain = []

    print('Start sampling with sampler:', samp, ' Total iterations:', args.n_steps)


    for i in range(args.n_steps):

        if i%1000==0:
            print(i)

        if (sampler.entropy == False):
            if samp!='acs' and samp!='pt+dmala':
                xhat = sampler.step(x.detach(), model).detach()
            else:
                xhat = sampler.step(x.detach(), model,i).detach()

        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            #print(xhat)
            xahat = sample_tuple[1].detach()

        x = xhat
        if (sampler.entropy == True):
            x_a = xahat


        x_list=x[0].tolist()
        if x_list not in chain and len(set(x_list))==DATA_DIM and i>=2000:
            print("yay!",len(chain))
            chain.append(x_list)

    print(chain, len(chain))
    one_hot_chain = to_one_hot_batch(chain, DATA_DIM).to(device)
    avg_cost, std_cost, max_cost, min_cost, median_cost, range_cost  = cost(one_hot_chain,model.city_coords)

    print(f"\nAverage Cost of Sampled Tours: {avg_cost:.4f}")
    print(f"Standard Deviation of Tour Costs: {std_cost:.4f}")
    print(f"\nMin Cost of Sampled Tours: {min_cost:.4f}")
    print(f"Max Deviation of Tour Costs: {max_cost:.4f}")
    print(f"\nMedian Cost of Sampled Tours: {median_cost:.4f}")
    print(f"Range Deviation of Tour Costs: {range_cost:.4f}")

    SD = []
    SD1=[]
    chain_paths=chain


    from itertools import combinations

    for path, path1 in combinations(chain_paths, 2):  # Generates unique pairs (path, path1)
        SD.append(pairwise_mismatch_count(path, path1))
        SD1.append(jaccard_similarity(path, path1))

    print(np.mean(SD), np.std(SD))
    print(np.mean(SD1), np.std(SD1))


if __name__ == "__main__":
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    import matplotlib.pyplot as plt
    import math
    from scipy.stats import entropy

    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str, default='edula')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--step_size_a', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=1)
    parser.add_argument('--tempchains', type=int, default=10)
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--cities', type=int, default=5)
    parser.add_argument('--G', type=int, default=10)
    parser.add_argument('--L', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default="./data")

    parser.add_argument("--num_cycles", type=int, default=250)
    parser.add_argument("--burnin_budget", type=int, default=200)
    parser.add_argument("--burnin_big_bal", type=float, default=.95)
    parser.add_argument("--burnin_small_bal", type=float, default=.5)
    parser.add_argument("--a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--bal_resolution", type=int, default=10)
    parser.add_argument("--initial_balancing_constant", type=float, default=.5)
    parser.add_argument("--big_step", type=float, default=2.0)
    parser.add_argument("--use_manual_EE", action="store_true")
    parser.add_argument("--adapt_every", type=int, default=25)
    parser.add_argument("--big_step_sampling_steps", type=int, default=5)
    parser.add_argument("--small_step", type=float, default=0.2)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=1.0)
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--min_lr", default=.011, type=float)

    args = parser.parse_args()

    main(args)
