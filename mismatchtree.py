def mismatchTree(s, k):
    """
    Builds mismatch tree for a string s up to k mismatches
    """
    if len(s)==1 and k==0: # last letter no more mismatch available
        return [s]
    elif len(s)==1 and k>0: # last letter but mismatch still available
        return ['T', 'A', 'C', 'G']
    elif len(s) > 1 and k==0: # not last letter and no more mismatch available
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
    Tests if a string s belongs to a mismatch_tree
    """
    current_sub_tree = mismatch_tree.copy()
    
    for letter in s:
        if isinstance(current_sub_tree, list): # i.e. leaf
            if letter in current_sub_tree:
                return True
            else:
                return False
        
        else: # i.e. inner node
            if letter in current_sub_tree.keys():
                current_sub_tree = current_sub_tree[letter]
            else:
                return False

def inMismatchTree(mismatch_tree, keys):
    return [key for key in keys if isInMismatchTree(mismatch_tree, key)]
           