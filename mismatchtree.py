def mismatchTree(s, k):
    """
    Recursive function to build mismatch tree for a string s up to k mismatches
    """
    if len(s)==1 and k==0:
        return [s]
    elif len(s)==1 and k>0:
        return ['T', 'A', 'C', 'G']
    elif len(s) > 1 and k==0:
        return {s[0]:mismatchTree(s[1:], 0)}
    
    else:
        d = {}
        for nucleotide in ['T', 'A', 'C', 'G']:
            if nucleotide == s[0]: # i.e. no mismatch at this node
                d[nucleotide] = mismatchTree(s[1:], k)
            else: # insert mismatch
                d[nucleotide] = mismatchTree(s[1:], k-1)
    return d

def isInMismatchTree(mismatch_tree, s):
    """
    Tests if a string s belongs to a given mismatch_tree
    """
    current_sub_tree = mismatch_tree
    n = len(s)-1
    compt = 0
    
    for letter in s:
        #if isinstance(current_sub_tree, list): # i.e. we attained a set of leaves
        if compt == n:
            return letter in current_sub_tree
        else: # i.e. inner node
            compt += 1
            if letter in current_sub_tree:
                current_sub_tree = current_sub_tree[letter]
            else:
                return False

def inMismatchTree(mismatch_tree, list_of_strings):
    "Given a list of string, determines all elements of the list which belong to a given mismatch tree"
    return [s for s in list_of_strings if isInMismatchTree(mismatch_tree, s)]
           