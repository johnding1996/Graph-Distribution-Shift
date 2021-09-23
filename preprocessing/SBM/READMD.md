# SBM Synthetic Dataset Proposals

For SBN datasets, the proposed domain variation type is the composition of graph communities (subgraphs).

Consider we have $r$ SBM communities in total.

The with-in-community edge probability $p_i$ characterize a specific community $i$, for $i=1,...,r$.

We want the SBM to be strongly assortative to make the community detection problem theoretically easier. In this sense, we consider $p_r > p_{r-1} > ... > p_1 > q$. 

Now we try do define the domain variation as the composition of (all or some) communities in the graphs. We select $s$ out of $r$ communities in each graph.

### Benchmark Idea 1: SBM-Isolation

Here we consider the domain variation is defined by random selection of $s$ communities from the complete set of $r$ communities. There will be ${r \choose s}$ domains.

Now if this is fixed, we can no longer change the with-in-community edge probabilities of the selected communities.

What we can change now is either the out-of-community edge probability or the size of each community. And by varying those selected communities, the task could only be predicting how many communities out of the $s$ ones are isolated. However, varying the sizes will affect the graph size and would be easy to predict. We consider varying each community by changing the out-out-community edge probabiity to $q' < q$.

Hyper-parameter setting: $r=9$, $s=7$. So there are 36 domains. And $p_9 = 0.9, p_8 = 0.85, ..., p_1=0.5$, $q=0.4$, $q'=0.3$. There will be 8 classes with label $0, ..., 7$ being the numberof isolated communities.

### Benchmark Idea 2: SBM-Environment

Here we first split the communities into two subsets, with $r_1$ and $r_2$ communities respectively. We random select $s_1$ and $s_2$ communities from two subsets respectively, and there will be $s=s_1+s_2$ communities at last again. We consider the selection process from the first subset as the domain variation, and from the second subset as the label. Thus there will be ${r_1 \choose s_1}$ domains and ${r_2 \choose s_2}$ labels.

We hope the communities in the two sets are similar, so the distribution of their with-in-community edge probabilites are around the same. For example, if $r_1 = r_2 +/- 1$, we can make the $p_i$'s from two sets next two each other in the order list $p_r > p_{r-1} > ... > p_1$.

Hyper-parameter setting: $r=11=6+5=r_1+r_2$, $s=7=4+3=s_1+s_2$, so there are 15 domains and 10 classes. We set $p_11 = 0.9, p_10 = 0.85, ..., p_1=0.4$, $q=0.3$. And the first set of communities has $p_11, p_9, ..., p_1$, the second set has the rest.


### Other Comments:
As we see there will always be $s=7$ communities at last in a graph. We set the size of each community to be a random number from 5 (inclusive) to 35 (exclusive). Thus the total size will be in range ${35, ..., 238}$. The node feature will always be i.i.d. random ints in set ${0, 1, 2}$. There is only 1 dimension of node feature. These setups are the same as in "Benchmarking GNNs" paper.