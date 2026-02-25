import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
#from tensorflow.python.framework.device import check_valid
import utils
import math
from tuning_components import *

class GWG_TSP(nn.Module):
    def __init__(self, dim, num_cls, n_steps=10, temp=1.0, step_size=0.2):
        super().__init__()
        self.dim = dim  # Number of cities
        self.n_steps = n_steps  # Number of Gibbs sampling steps
        self.temp = temp  # Temperature for MH acceptance
        self.step_size = step_size
        self.num_cls = num_cls  ### number of classes in each dimension
        self.entropy = False
        # Step size for gradient-based proposal

    def get_grad(self, x, model):
        """Compute the gradient of the tour cost w.r.t. the current state."""
        x = x.clone().detach().requires_grad_(True)
        cost = model.forward(x)

        if not cost.requires_grad:
            raise RuntimeError("Cost does not require gradients. Check differentiability.")

        cost.backward()
        return x.grad.clone().detach()

    def to_one_hot(self, x):
        """Convert a city permutation to one-hot encoding."""
        one_hot = torch.zeros((x.shape[0], self.dim, self.dim)).to(x.device)
        one_hot.scatter_(2, x.unsqueeze(2), 1.0)
        return one_hot

    def step(self, x, model):
        """
        Perform gradient-informed Gibbs sampling for TSP.
        - x: Batch of current tours (batch_size x num_cities)
        - model: Energy function model (tour cost)
        """
        x_cur = x.clone()
        batch_size = x_cur.size(0)

        for _ in range(self.n_steps):


            x_cur_one_hot = self.to_one_hot(x_cur)
            grad = self.get_grad(x_cur_one_hot, model) / self.temp  # Shape: [batch_size, N, num_cls]

            for b in range(batch_size):
                # Select a city index (position in the sequence) to update
                city_idx = torch.randint(0, self.dim, (1,)).item()

                # Construct logits for selecting a new city using the gradient
                logits = grad[b, city_idx]  # Shape: [num_cls]

                # Gibbs sampling: propose a new city at this position
                proposal_dist = dists.Categorical(logits=logits)
                new_city = proposal_dist.sample()

                # Replace city at selected position
                x_new = x_cur[b].clone()
                x_new[city_idx] = new_city  # Replace instead of swapping

                # One-hot encode the proposed tour
                x_new_one_hot = self.to_one_hot(x_new.unsqueeze(0))

                # Compute forward and reverse probabilities
                #x_cur_one_hot = self.to_one_hot(x_cur)

                cost_cur = model.forward(x_cur_one_hot).squeeze()
                cost_new = model.forward(x_new_one_hot).squeeze()

                forward_prob = proposal_dist.log_prob(new_city)

                grad_new = self.get_grad(x_new_one_hot, model) / self.temp
                reverse_logits = grad_new[0, city_idx ]
                reverse_dist = dists.Categorical(logits=reverse_logits)
                reverse_prob = reverse_dist.log_prob(x_cur[b][city_idx].to(x.device))

                # Compute Metropolis-Hastings acceptance probability
                mh_acceptance = torch.exp((cost_new - cost_cur) + reverse_prob - forward_prob)

                # Accept or reject the proposal
                if torch.rand(1).item() < mh_acceptance.item():
                    x_cur[b] = x_new  # Accept the new tour

        # Final validity check
        x_cur = x_cur.long()
        if torch.unique(x_cur).numel() != self.dim:  # Ensure a valid TSP sequence
            x_cur = x  # Revert to previous valid state

        return x_cur



class ELangevin_TSP(nn.Module):
    def __init__(self, dim, num_cls=3, n_steps=10, multi_hop=False, temp=2., step_size=0.2, step_size_a=0.1, eta=1, mh=True, device=None):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.alpha_a=step_size_a
        self.eta=eta# rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2

        self.mh = mh
        self.num_cls = num_cls  ### number of classes in each dimension
        self.entropy = True

    def get_grad(self, x, model):
        x = x.clone().detach().requires_grad_(True)

        # Compute the cost
        cost = model.forward(x)

        # Check if cost requires gradient
        if not cost.requires_grad:
            raise RuntimeError("Cost does not require gradients. Check differentiability.")

        # Backpropagate
        cost.backward()

        # Extract the gradients
        gradients = x.grad.clone().detach()

        return gradients

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def step(self, x, x_1, model):
        '''
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        '''
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []
        x_a=x_1
        #print(self.eta)

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_one_hot = self.to_one_hot(x_cur)
            # x_cur_1=x_cur

            grad = self.get_grad(x_cur_one_hot, model)/ self.temp  # Shape: [52, 2]
            # print(grad)
            grad_cur_a = (1 / (self.eta)) * (x_cur_one_hot - x_a)
            #print(grad_cur_a)
            grad = grad - grad_cur_a/self.temp
            """
            grad_cur = grad[0:1, range(self.dim), x_cur[0, :]]  # Extract gradients of current positions
            first_term = grad.detach().clone() - grad_cur.unsqueeze(2).repeat(1, 1,
                                                                              self.num_cls)  # Correct broadcasting
            """
            grad_cur = grad.gather(2, x_cur.unsqueeze(-1))  # Safe indexing

            first_term = grad - grad_cur.expand(-1, -1, self.num_cls)


            # Compute second term (prevent self-transitions)
            second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
            second_term[0, range(self.dim), x_cur[0, :]] = 0.  # Ensure no self-transitions

            # Sample new state
            #logits_c=first_term - second_term
            logits_c = (first_term - second_term) / (torch.std(first_term - second_term, dim=-1, keepdim=True) + 1e-8)
            """
            raw = first_term - second_term
            print("eta", self.eta,
                  "raw std", raw.std().item(),
                  "raw meanabs", raw.abs().mean().item(),
                  "logits std", logits_c.std().item(),
                  "logits meanabs", logits_c.abs().mean().item())

            diff = (x_cur_one_hot - x_a)
            print("eta", self.eta,
                  "||diff||", diff.norm().item(),
                  "||grad_cur_a||", grad_cur_a.norm().item(),
                  "max|diff|", diff.abs().max().item(),
                  "max|grad_cur_a|", grad_cur_a.abs().max().item())
            """


            cat_dist = torch.distributions.categorical.Categorical(logits=logits_c)
            x_delta = cat_dist.sample()  # Shape: [1, N]

            x_delta_a = x_a + (self.alpha_a / 2.) * grad_cur_a + (
                    (1 * self.alpha_a) ** 0.5) * torch.randn_like(x_cur_one_hot, device=x_a.device)

            # Validity check: ensure unique cities in the tour
            #print(x_delta,x_delta_a)

            if torch.unique(x_delta).numel() != self.dim:
                x_delta = x_cur  # Reject invalid state
                #print("!")
                x_delta_a = x_a


            if self.mh and torch.equal(x_delta,x_cur)==False:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta)
                grad_d = self.get_grad(x_delta_one_hot, model) / self.temp

                grad_delta_a = (1 / (self.eta )) * (x_delta_one_hot - x_delta_a)

                grad_d=grad_d-grad_delta_a/self.temp

                grad_del = grad_d.gather(2, x_delta.unsqueeze(-1))  # Safe indexing


                first_term_delta = grad_d - grad_del.expand(-1, -1, self.num_cls)  # Correct broadcasting
                """
                grad_del = grad_d[0:1, range(self.dim), x_delta[0, :]]  # Extract gradients of current positions
                first_term = grad_d.detach().clone() - grad_del.unsqueeze(2).repeat(1, 1,
                                                                                  self.num_cls)  # Correct broadcasting
                """
                # Compute second term (prevent self-transitions)
                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / self.step_size
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.  # Ensure no self-transitions

                # Reverse proposal distribution for MH acceptance

                logits_d = (first_term_delta - second_term_delta) / (torch.std(first_term_delta - second_term_delta, dim=-1, keepdim=True) + 1e-8)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_d)
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)  # Reverse proposal log prob

                m_term = (model(x_delta).squeeze() - torch.sum(
                    torch.sum(torch.pow(x_delta_one_hot - x_delta_a, 2), dim=1)) / (
                                  2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(
                             torch.sum(torch.pow(x_cur_one_hot - x_a, 2), dim=1)) / (
                                         2 * self.eta))

                additional_m_term_reverse = torch.sum(torch.sum(
                    torch.pow(x_a - (x_delta_a + (self.alpha_a / 2) * grad_delta_a), 2), dim=1)) / (
                                                    -2 * self.alpha_a)
                additional_m_term_forward = torch.sum(torch.sum(
                    torch.pow(x_delta_a - (x_a + (self.alpha_a / 2) * grad_cur_a), 2), dim=1)) / (
                                                    -2 * self.alpha_a)

                la = m_term + additional_m_term_reverse - additional_m_term_forward + lp_reverse - lp_forward

                # Perform MH acceptance step
                a = (la.exp() > torch.rand_like(la)).float()  # Accept/Reject

                # Track acceptance rate
                a_s.append(a.mean().item())

                # Accept or reject the proposed state
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                x_a = x_delta_a * a[:, None] + x_a * (1. - a[:, None])

            else:
                x_cur = x_delta
                x_a = x_delta_a

            x_cur = x_cur.long()
            #x_a = x_a.long()

        return x_cur, x_a

class Langevin_TSP(nn.Module):
    def __init__(self, dim, num_cls=3, n_steps=10, multi_hop=False, temp=2., step_size=0.2, mh=True, device=None):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2

        self.mh = mh
        self.num_cls = num_cls  ### number of classes in each dimension
        self.entropy = False

    def get_grad(self, x, model):
        x = x.clone().detach().requires_grad_(True)

        # Compute the cost
        cost = model.forward(x)

        # Check if cost requires gradient
        if not cost.requires_grad:
            raise RuntimeError("Cost does not require gradients. Check differentiability.")

        # Backpropagate
        cost.backward()

        # Extract the gradients
        gradients = x.grad.clone().detach()

        return gradients

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def step(self, x, model):
        '''
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        '''
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_one_hot = self.to_one_hot(x_cur)
            # x_cur_1=x_cur

            grad = self.get_grad(x_cur_one_hot, model)/ self.temp  # Shape: [52, 2]
            # print(grad)
            """
            grad_cur = grad[0:1, range(self.dim), x_cur[0, :]]  # Extract gradients of current positions
            first_term = grad.detach().clone() - grad_cur.unsqueeze(2).repeat(1, 1,
                                                                              self.num_cls)  # Correct broadcasting
            """
            grad_cur = grad.gather(2, x_cur.unsqueeze(-1))  # Safe indexing
            first_term = grad - grad_cur.expand(-1, -1, self.num_cls)

            # Compute second term (prevent self-transitions)
            second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
            second_term[0, range(self.dim), x_cur[0, :]] = 0.  # Ensure no self-transitions

            # Sample new state
            #logits=first_term - second_term
            logits_c = (first_term - second_term) / (torch.std(first_term - second_term, dim=-1, keepdim=True) + 1e-8)
            """
            print("Logits Mean:", logits.mean().item())
            print("Logits Std:", logits.std().item())
            print("Sampling Probabilities:", torch.softmax(logits, dim=-1))
            """

            cat_dist = torch.distributions.categorical.Categorical(logits=logits_c)
            x_delta = cat_dist.sample()  # Shape: [1, N]

            # Validity check: ensure unique cities in the tour
            #print(x_delta)

            if torch.unique(x_delta).numel() != self.dim:
                x_delta = x_cur  # Reject invalid state
                #print("!")


            if self.mh and torch.equal(x_delta,x_cur)==False:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta)
                grad_d = self.get_grad(x_delta_one_hot, model) / self.temp

                grad_del = grad_d.gather(2, x_delta.unsqueeze(-1))  # Safe indexing
                first_term_delta = grad_d - grad_del.expand(-1, -1, self.num_cls)  # Correct broadcasting
                """
                grad_del = grad_d[0:1, range(self.dim), x_delta[0, :]]  # Extract gradients of current positions
                first_term = grad_d.detach().clone() - grad_del.unsqueeze(2).repeat(1, 1,
                                                                                  self.num_cls)  # Correct broadcasting
                """
                # Compute second term (prevent self-transitions)
                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / self.step_size
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.  # Ensure no self-transitions

                # Reverse proposal distribution for MH acceptance

                logits_d = (first_term_delta - second_term_delta) / (torch.std(first_term_delta - second_term_delta, dim=-1, keepdim=True) + 1e-8)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_d)
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)  # Reverse proposal log prob

                # Compute the MH acceptance ratio
                m_term = (model.forward(x_delta_one_hot).squeeze() - model.forward(x_cur_one_hot).squeeze())
                #print(m_term)# Energy difference
                log_acceptance_ratio = m_term + lp_reverse - lp_forward  # Log-space acceptance ratio

                # Perform MH acceptance step
                a = (log_acceptance_ratio.exp() > torch.rand_like(log_acceptance_ratio)).float()  # Accept/Reject

                # Track acceptance rate
                a_s.append(a.mean().item())

                # Accept or reject the proposed state
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

            x_cur = x_cur.long()
        return x_cur


class HiSS_ScoreSamp_TSP(nn.Module):
    def __init__(self, dim, n_steps=10, num_cls=1, approx=False, multi_hop=False, fixed_proposal=False, temp=2.,
                 step_size=0.1,mh=True, eta=10 ** 4):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.eta = eta  # flatness
        self.entropy = True

        self.num_cls = num_cls
        self.mh = mh
        self.a_s = []
        self.hops = []

    def get_grad(self, x, model):
        x = x.clone().detach().requires_grad_(True)

        cost = model.forward(x)

        # Check if cost requires gradient
        if not cost.requires_grad:
            raise RuntimeError("Cost does not require gradients. Check differentiability.")

        cost.backward()

        gradients = x.grad.clone().detach()
        return gradients

    def to_one_hot(self, x):

        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)

        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def step(self, x, x_a, model):
        #print("inside score sampler")

        x_cur = x
        x_a_cur = x_a

        EPS = 1e-10
        for i in range(self.n_steps):
            #print("small")
            """
            print(i)
            print(x_cur)
            """
            x_cur_one_hot = self.to_one_hot(x_cur)
            grad_x_cur = self.get_grad(x_cur_one_hot, model)/ self.temp
            grad = grad_x_cur + (torch.tanh((x_a_cur - x_cur_one_hot) / 2 * self.eta) / (self.eta * self.temp))

            grad_cur = grad.gather(2, x_cur.unsqueeze(-1))  # Safe indexing
            first_term = grad - grad_cur.expand(-1, -1, self.num_cls)


            second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
            second_term[0, range(self.dim), x_cur[0, :]] = 0.  # En # Prevent staying in the same state

            logits_c = (first_term - second_term) / (torch.std(first_term - second_term, dim=-1, keepdim=True) + EPS)

            # Sample new state
            cat_dist = torch.distributions.categorical.Categorical(logits=logits_c)
            x_delta = cat_dist.sample()  # Shape: [1, N]

            #print(x_delta)

            if torch.unique(x_delta).numel() != self.dim:
                x_delta = x_cur  # Reject invalid state


            if self.mh and torch.equal(x_delta, x_cur)==False:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta)

                grad_d = self.get_grad(x_delta_one_hot, model) / self.temp

                grad_delta = grad_d + (
                        torch.tanh((x_a_cur - x_delta_one_hot) / 2 * self.eta) / (self.eta * self.temp))

                grad_delta_cur = grad_delta.gather(2, x_delta.unsqueeze(-1))  # Safe indexing
                first_term_delta = grad_delta - grad_delta_cur.expand(-1, -1, self.num_cls)

                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / self.step_size
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.

                logits_d = (first_term_delta - second_term_delta) / (torch.std(first_term_delta - second_term_delta, dim=-1, keepdim=True) + EPS)

                # Reverse proposal distribution for MH acceptance
                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_d)

                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)  # Reverse proposal log prob

                # Compute the MH acceptance ratio
                m_term = (model.forward(x_delta_one_hot).squeeze() - 2 *torch.sum( (
                    torch.log(torch.cosh((x_a_cur - x_delta_one_hot) / 2 * self.eta)).sum(dim=1)))) - (
                                 model.forward(x_cur_one_hot).squeeze() - 2 *torch.sum( (
                             torch.log(torch.cosh((x_a_cur - x_cur_one_hot) / 2 * self.eta)).sum(dim=1))))

                """
                print(m_term, m_term.shape)
                print(lp_reverse, lp_reverse.shape)
                print(lp_forward, lp_forward.shape)
                """

                la = m_term + lp_reverse - lp_forward  # Log-space acceptance ratio

                # Perform MH acceptance step
                a = (la.exp() > torch.rand_like(la)).float()  # Accept/Reject

                # Track acceptance rate
                self.a_s.append(a.mean().item())
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                x_cur = x_cur.squeeze(1)

            else:
                x_cur = x_delta

            x_cur = x_cur.long()



        """
        print("score-samp-end")
        print(x_cur)
        """
        return (x_cur, x_a_cur)  # Update x, not x_a


class HiSS_TSP(nn.Module):
    def __init__(self, dim, num_cls=5, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2.,
                 step_size=0.2,
                 mh=True, eta=10 ** 4, score_sweeps=50, n=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.chains = n
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.eta = eta  # flatness
        self.entropy = True

        self.num_cls = num_cls  ### number of classes in each dimension

        self.L = score_sweeps
        self.mh = mh
        self.score_based_sampler = HiSS_ScoreSamp_TSP(self.dim, num_cls=self.num_cls, n_steps=self.L,
                                                                 fixed_proposal=False, approx=True,
                                                                 multi_hop=False,
                                                                 temp=2., step_size=self.step_size, eta=self.eta, mh=True)
        self.a_s = []
        self.hops = []

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def enforce_valid_tour_minimal(sself,x_delta, num_cities):
        """
        Enforces that x_delta is a valid TSP tour by making minimal swaps to replace duplicate cities with missing ones.
        """
        batch_size, dim = x_delta.shape

        for b in range(batch_size):
            unique_cities, counts = x_delta[b].unique(return_counts=True)
            missing_cities = list(set(range(num_cities)) - set(unique_cities.tolist()))

            if not missing_cities:
                continue  # Already valid

            # Identify duplicate cities
            duplicate_indices = (counts > 1).nonzero(as_tuple=True)[0]

            # Replace duplicates with missing cities using minimal swaps
            for idx in duplicate_indices:
                duplicates = (x_delta[b] == unique_cities[idx]).nonzero(as_tuple=True)[0]
                for d in duplicates[1:]:  # Keep one, replace the rest
                    if missing_cities:
                        x_delta[b, d] = missing_cities.pop(0)

        return x_delta

    def logistic_denoise(self, x_a_cur, eta):
        """
        Logistic denoising followed by minimal corrections to ensure a valid TSP tour.
        """
        batch_size, dim, num_cls = x_a_cur.shape
        EPS = 1e-10

        # Generate all possible one-hot encoded vectors for each city
        all_one_hot_states = torch.eye(num_cls, device=x_a_cur.device).expand(batch_size, dim, num_cls, num_cls)

        # Expand x_a_cur to match the shape for broadcasting
        x_a_expanded = x_a_cur.unsqueeze(-2)

        # Compute energy differences
        diff = (x_a_expanded - all_one_hot_states) / (2 * eta)
        energy = -2 * torch.log(torch.cosh(diff) + EPS)

        # Sum across candidate states and normalize
        energy_sum = energy.sum(dim=-2)

        # Normalize logits before softmax
        #energy_sum = (energy_sum - energy_sum.mean(dim=-1, keepdim=True)) / (energy_sum.std(dim=-1, keepdim=True) + EPS)

        # Convert energy into probabilities
        probs = torch.softmax(energy_sum, dim=-1)

        # Sample a city for each position based on the computed probabilities
        dist = torch.distributions.Categorical(probs)
        x_delta = dist.sample()
        #print("before", str(x_delta))

        # Apply minimal corrections for a valid TSP tour
        if torch.unique(x_delta).numel() != self.dim:
            x_delta = self.enforce_valid_tour_minimal(x_delta, num_cls)
        #print("after", str(x_delta))

        return x_delta.long()

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a
        EPS = 1e-10
        #print("Gibbs-Sweep")

        for i in range(self.n_steps):
            #print("big")
            """
            print(i)
            print(x_cur)
            """
            x_cur_one_hot = self.to_one_hot(x_cur)

            uniform = torch.rand_like(x_cur_one_hot)
            logistic_noise = torch.log(uniform / (1 - uniform))
            x_a_cur = x_cur_one_hot + self.eta * logistic_noise
            #print(x_a_cur)
            x_delta = self.logistic_denoise(x_a_cur, self.eta)
            """
            print("after logistic denoising")
            print(x_delta)
            """

            if torch.unique(x_delta).numel() != self.dim:
                x_delta = x_cur  # Reject invalid state

            if self.mh and torch.equal(x_delta, x_cur)==False:
                q_forward = (x_a_cur - x_cur_one_hot) / (2 * self.eta)
                lp_forward = -2*torch.sum(( torch.log(torch.cosh(q_forward) + EPS).sum(dim=1)))

                x_delta_one_hot = self.to_one_hot(x_delta)
                q_reverse = (x_a_cur - x_delta_one_hot) / (2 * self.eta)
                lp_reverse = -2 *torch.sum((torch.log(torch.cosh(q_reverse) + EPS).sum(dim=1)))


                m_term = (model.forward(x_delta_one_hot).squeeze() - 2*torch.sum( torch.log(
                    torch.cosh((x_a_cur - x_delta_one_hot) / (2 * self.eta))).sum(dim=1))) - (
                                 model.forward(x_cur_one_hot).squeeze() - 2 *torch.sum(torch.log(
                             torch.cosh((x_a_cur - x_cur_one_hot) / (2 * self.eta))).sum(
                             dim=1)))


                la = m_term + lp_reverse - lp_forward
                #print(m_term, lp_reverse, lp_forward)

                a = (la.exp() > torch.rand_like(la)).float()
                #print(a)

                self.a_s.append(a.mean().item())
                x_cur = x_delta * a + x_cur * (1. - a)
                x_cur = x_cur.squeeze(1)

            else:
                x_cur = x_delta

            x_cur = x_cur.long()
            """
            print("after mh step")
            print(x_cur)
            """

            t = self.score_based_sampler.step(x_cur.detach(), x_a_cur.detach(), model)
            x_cur = t[0].detach()

        return (x_cur, x_a_cur)



class PT_DMALA_TSP(nn.Module):
    def __init__(self, dim,num_cls=1, n_steps=10, n_chains=4, temps=None, step_size=0.2, mh=True, swap_interval=5):
        super().__init__()
        self.dim = dim
        self.num_cls = num_cls
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.temps = temps if temps else [1.0] + [10** i for i in range(1, n_chains)]  # Default temperatures
        #self.temps = torch.logspace(0, 1, self.n_chains).tolist()
        self.step_size = step_size
        self.mh = mh
        self.swap_interval = swap_interval
        self.entropy = False

        # Initialize Langevin samplers for each temperature
        self.samplers = [Langevin_TSP(self.dim,self.num_cls, self.n_steps,temp=2 * temp, step_size=self.step_size, mh=self.mh) for temp in self.temps]
    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def parallel_tempering_step(self, x, model, step_count):
        """
        Perform one step of parallel tempering for all chains and temperatures.
        :param x: Tensor of shape (n_chains, K, dim), where K is the number of temperatures.
        :param model: Energy function to evaluate the samples.
        :param step_count: Current iteration count.
        :return: Updated tensor of shape (n_chains, K, dim).
        """
        n_chains, K, dim = x.shape

        # Step 1: Perform Langevin sampling for all chains and temperatures
        new_samples = []
        for temp_idx, sampler in enumerate(self.samplers):
            temp_samples = sampler.step(x[:, temp_idx, :], model)  # Batched sampling for all chains at this temperature
            new_samples.append(temp_samples)
        new_samples = torch.stack(new_samples, dim=1)  # Shape: (n_chains, K, dim)

        # Step 2: Perform swaps between adjacent temperatures if needed
        if step_count % self.swap_interval == 0:
            beta = torch.tensor([1.0 / temp for temp in self.temps], device=x.device)  # Shape: (K,)
            for i in range(K - 1):
                x_i, x_j = new_samples[:, i, :], new_samples[:, i + 1, :]
                beta_i, beta_j = beta[i], beta[i + 1]
                if torch.equal(x_i, x_j)==False:
                    x_i_oh=self.to_one_hot(x_i)
                    x_j_oh=self.to_one_hot(x_j)
                    delta_energy = (beta_i - beta_j) * (model.forward(x_i_oh).squeeze() - model.forward(x_j_oh).squeeze())
                    #print(delta_energy)
                    acceptance_probs = torch.exp(delta_energy).squeeze()

                    # Randomly accept/reject swaps
                    swap_mask = (torch.rand(n_chains, device=x.device) < acceptance_probs).float()
                    swap_mask = swap_mask.unsqueeze(1)  # Shape: (n_chains, 1)
                    swapped_i = swap_mask * x_j + (1 - swap_mask) * x_i
                    swapped_j = swap_mask * x_i + (1 - swap_mask) * x_j
                    new_samples[:, i, :] = swapped_i
                    new_samples[:, i + 1, :] = swapped_j

        return new_samples

    def step(self, x, model, step_count):
        """
        Perform parallel tempering for all independent chains.
        :param x: Tensor of shape (n_chains, dim), where n_chains is the batch size.
        :param model: Energy function to evaluate the samples.
        :param step_count: Current iteration count.
        :return: Tensor of updated samples (n_chains, dim) with only T=1 samples for each chain.
        """
        n_chains, dim = x.shape
        K = len(self.temps)

        # Step 1: Reshape x into individual chains with K temperatures
        x = x.unsqueeze(1).repeat(1, K, 1)  # Shape: (n_chains, K, dim)

        # Step 2: Perform parallel tempering
        updated_chains = self.parallel_tempering_step(x, model, step_count)

        # Step 3: Extract T=1 samples for each chain (first temperature)
        return updated_chains[:, 0, :]  # Shape: (n_chains, dim)


class ACS_TSP(nn.Module):
    def __init__(
            self,
            dim,
            device,
            num_cls,
            num_cycles,
            num_iters,
            n_steps=10,
            approx=False,
            multi_hop=False,
            fixed_proposal=False,
            temp=1.0,
            mean_stepsize=0.2,
            mh=True,
            initial_balancing_constant=1,
            burnin_adaptive=False,
            burnin_budget=500,
            burnin_lr=0.5,
            sbc=False,
            big_step=None,
            big_bal=None,
            small_step=None,
            small_bal=None,
            iter_per_cycle=None,
            min_lr=None,
            a_s_cut=None,
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_steps = n_steps
        self.num_cls = num_cls
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = mean_stepsize
        self.initial_step_size = mean_stepsize
        self.num_cycles = num_cycles
        self.num_iters = num_iters
        self.burnin_adaptive = burnin_adaptive
        self.burnin_budget = burnin_budget
        self.burnin_lr = burnin_lr
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m)
        self.mh = mh
        self.a_s = []
        self.hops = []
        self.initial_balancing_constant = initial_balancing_constant
        self.balancing_constant = initial_balancing_constant
        self.bal = initial_balancing_constant
        if iter_per_cycle and sbc:
            self.iter_per_cycle = iter_per_cycle
        else:
            print(self.num_iters)
            print(self.num_cycles)
            self.iter_per_cycle = math.ceil(self.num_iters / self.num_cycles)
        self.step_sizes = self.calc_stepsizes(self.step_size)
        self.diff_values = []
        self.flip_probs = []
        self.D = None
        self.balancing_constants = self.calc_balancing_constants(self.initial_balancing_constant)
        if sbc and big_step and big_bal and small_step and small_bal:
            self.sbc = sbc
            self.big_step = big_step
            self.small_step = small_step
            self.big_bal = big_bal
            self.small_bal = small_bal
            self.step_sizes = [big_step] + [small_step] * (self.iter_per_cycle - 1)
            self.balancing_constants = [big_bal] + [small_bal] * (
                    self.iter_per_cycle - 1
            )

        else:
            self.sbc = False
        self.min_lr = min_lr
        if self.min_lr:
            self.min_lr_cutoff()
        self.a_s_cut = a_s_cut
        self.track_terms = False
        self.t1 = None
        self.t2 = None
        self.counts = 0
        self.entropy=False



    def get_name(self):
        if self.mh:
            base = "acs"
        else:
            base = "acs_no_mh"
        if self.burnin_adaptive:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_budget_{self.burnin_budget}_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
            )
        else:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_stepsize_{self.initial_step_size}_initbal_{self.balancing_constant}"
            )
        if self.sbc:
            if self.burnin_adaptive:
                name = base + f"_adaptive_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
            else:
                name = (
                        base
                        + f"_cycle_length_{self.iter_per_cycle}_big_s_{self.big_step}_b_{self.big_bal}_small_s_{self.small_step}_b_{self.small_bal}"
                )
        if self.min_lr:
            name += f"_min_lr_{self.min_lr}"
        return name

    def calc_balancing_constants(self, init_bal):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_balancing_constant = (init_bal - 0.5) / 2 * (np.cos(inner)) + (
                    init_bal + 0.5
            ) / 2
            res.append(cur_balancing_constant)
        res = torch.tensor(res, device=self.device)
        return res

    def min_lr_cutoff(self):
        if self.mh:
            min_step = 0.2
        else:
            min_step = 0.1
        for i in range(len(self.step_sizes)):
            if self.step_sizes[i] <= min_step:
                self.step_sizes[i] = min_step
                self.balancing_constants[i] = 0.5

    def calc_stepsizes(self, mean_step):
        #print("inside calc_stepsizes")
        #print(mean_step)
        res = []
        total_iter = self.iter_per_cycle
        #print(total_iter)
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = torch.tensor(res, device=self.device)
        #print(res)
        return res

    def adapt_big_step(
            self,
            x_init,
            model,
            budget,
            init_big_step,
            init_big_bal,
            lr,
            test_steps,
            a_s_cut,
            use_dula,
            bdmala=None,
    ):
        if bdmala is None:
            bdmala = Langevin_TSP(
                dim=self.dim,
                num_cls=self.num_cls,
                n_steps=self.n_steps,
                approx=self.approx,
                multi_hop=self.multi_hop,
                fixed_proposal=self.fixed_proposal,
                step_size=1,
                mh=True,
                bal=init_big_bal,
            )

        (
            x_cur,
            alpha_max,
            alpha_max_metrics,
            _,
        ) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget,
            init_step_size=init_big_step,
            x_init=x_init,
            use_dula=use_dula,
            lr=lr,
        )
        self.step_sizes[0] = alpha_max
        self.balancing_constants[0] = init_big_bal
        return x_cur, alpha_max, alpha_max_metrics

    def adapt_small_step(
            self,
            x_init,
            model,
            budget,
            init_small_step,
            init_small_bal,
            lr,
            test_steps,
            a_s_cut,
            use_dula,
            bdmala=None,
    ):
        if bdmala is None:
            bdmala = Langevin_TSP(
                dim=self.dim,
                num_cls=self.num_cls,
                n_steps=self.n_steps,
                approx=self.approx,
                multi_hop=self.multi_hop,
                fixed_proposal=self.fixed_proposal,
                step_size=1,
                mh=True,
                bal=init_small_bal,
            )
        x_cur, alpha_min, alpha_min_metrics, _ = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=x_init,
            budget=budget,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
        )
        for i in range(1, len(self.step_sizes)):
            self.step_sizes[i] = alpha_min
            self.balancing_constants[i] = init_small_bal
        return x_cur, alpha_min, alpha_min_metrics

    def tuning_alg(
            self,
            dula_x_init,
            model,
            budget,
            init_big_step,
            init_small_step,
            init_big_bal=0.95,
            init_small_bal=0.5,
            a_s_cut=0.5,
            test_steps=10,
            lr=0.5,
            step_zoom_res=5,
            step_size_pair=None,
            x_init_to_use="bal",
            bal_resolution=3,
            use_bal_cyc=False,
            dula_burnin=50,
            acs_burnin=0
    ):

        bdmala = Langevin_TSP(
            dim=self.dim,
            num_cls=self.num_cls,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=1,
            mh=True,
            bal=init_small_bal,
        )
        # pre burn in
        # if self.norm_mterm:
    def get_grad(self, x, model):
        x = x.clone().detach().requires_grad_(True)

            # Compute the cost
        cost = model.forward(x)

            # Check if cost requires gradient
        if not cost.requires_grad:
            raise RuntimeError("Cost does not require gradients. Check differentiability.")

            # Backpropagate
        cost.backward()

            # Extract the gradients
        gradients = x.grad.clone().detach()

        return gradients

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

        bdmala.D = self.D
        bdmala.mh = False
        bdmala.step_size = 5
        bdmala.bal = init_big_bal
        bal_x_init = dula_x_init
        for i in range(dula_burnin):
            dula_x_init = bdmala.step(dula_x_init, model).detach()
        bdmala.mh = True
        self.mh = True
        for i in range(acs_burnin):
            dula_x_init = self.step(dula_x_init, model, i).detach()
        total_res = {}
        possible_x_inits = [dula_x_init]
        # estimating alpha min
        use_dula = not self.mh
        if bdmala.t1 is not None:
            t1 = bdmala.t1 / bdmala.counts
            t1 = t1.mean(axis=0)
            D = torch.diag(t1)
        alpha_min_x_init, alpha_min, alpha_min_metrics, itr = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=dula_x_init,
            budget=budget // 2,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
            use_dula=use_dula,
        )
        possible_x_inits.append(alpha_min_x_init)
        (alpha_max_x_init, alpha_max, alpha_max_metrics, itrr) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget // 2,
            init_step_size=init_big_step,
            x_init=alpha_min_x_init,
            use_dula=use_dula,
        )

        init_big_bal = bdmala.bal
        possible_x_inits.append(alpha_max_x_init)
        total_res["alpha_max_metrics"] = alpha_max_metrics
        total_res["alpha_min_metrics"] = alpha_min_metrics

        opt_steps = self.calc_stepsizes(alpha_max / 2)

        for i in range(len(opt_steps)):
            if opt_steps[i] < alpha_min:
                break
        bal_x_init, opt_bal, bal_metrics = estimate_opt_bal(
            model=model,
            bdmala=bdmala,
            x_init=dula_x_init,
            init_bal=init_big_bal,
            opt_steps=opt_steps[:i],
            est_resolution=bal_resolution,
            test_steps=test_steps,
            use_dula=use_dula,
            init_small_bal=init_small_bal
        )
        possible_x_inits.append(bal_x_init)

        while i < len(opt_steps):
            opt_steps[i] = alpha_min
            opt_bal.append(init_small_bal)
            i += 1
        self.balancing_constants = opt_bal
        self.step_sizes = opt_steps
        total_res["bal_metrics"] = bal_metrics
        print("step sizes: \n")
        print(self.step_sizes)
        print("\n")
        print("bal: \n")
        print(self.balancing_constants)
        return possible_x_inits[-1], total_res



    def step(self, x, model, k_iter, return_diff=False):
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        #print("step_size: ", self.step_sizes)
        balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]

        for i in range(self.n_steps):
            x_cur_one_hot = self.to_one_hot(x_cur)
            # x_cur_1=x_cur
            grad = self.get_grad(x_cur_one_hot, model) / self.temp
            grad_bal=grad*balancing_constant

            grad_cur = grad_bal.gather(2, x_cur.unsqueeze(-1))  # Safe indexing
            first_term = grad_bal - grad_cur.expand(-1, -1, self.num_cls)# Correct broadcasting

            # Compute second term (prevent self-transitions)
            second_term = torch.ones_like(first_term).to(x_cur.device) / step_size
            second_term[0, range(self.dim), x_cur[0, :]] = 0.  # Ensure no self-transitions


            #forward_delta = self.diff_fn(x_cur, model)


            if self.D is not None:
                term2 = self.D * second_term
            else:
                term2 = second_term  # for binary {0,1}, the L2 norm is always 1
            if self.track_terms:
                if self.t1 is None:
                    self.t1 = first_term
                else:
                    self.t1 += first_term
                self.counts += 1

            logits_c = (first_term - term2) / (torch.std(first_term - term2, dim=-1, keepdim=True) + EPS)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits_c)
            x_delta = cat_dist.sample()  # Shape: [1, N]

            # Validity check: ensure unique cities in the tour

            if torch.unique(x_delta).numel() != self.dim:
                x_delta = x_cur

            if self.mh and torch.equal(x_delta, x_cur)==False:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta)

                grad_delta = self.get_grad(x_delta_one_hot, model) / self.temp

                grad_delta_bal = grad_delta* balancing_constant

                grad_delta_cur = grad_delta_bal.gather(2, x_delta.unsqueeze(-1))  # Safe indexing
                first_term_delta = grad_delta_bal - grad_delta_cur.expand(-1, -1, self.num_cls)  # Correct broadcasting

                # Compute second term (prevent self-transitions)
                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / step_size
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.  # Ensure no self-transitions

                if self.D is not None:
                    term2 = self.D * second_term_delta
                else:
                    term2 = second_term_delta  # fo

                logits_d = (first_term_delta - term2) / (torch.std(first_term_delta - term2, dim=-1, keepdim=True) + EPS)


                # Reverse proposal distribution for MH acceptance
                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_d)
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model.forward(x_delta_one_hot).squeeze() - model.forward(x_cur_one_hot).squeeze()
                la = m_term + lp_reverse - lp_forward
                print(la)
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                probs_tmp = torch.minimum(
                    torch.ones_like(la, device=la.device), la.exp()
                )
                self.a_s.append(probs_tmp.detach().mean().item())
            else:
                x_cur = x_delta

            x_cur = x_cur.long()
        if return_diff:
            probs = torch.minimum(torch.ones_like(la, device=la.device), la.exp())
            return x_cur.detach(), first_term , probs
        else:
            return x_cur.detach()