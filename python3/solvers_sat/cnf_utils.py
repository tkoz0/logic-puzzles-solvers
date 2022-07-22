from typing import List, Tuple
import unittest

def cnf_stats(cnf: List[List[int]]) -> Tuple[int,int]:
    ''' get number of variables and clauses '''
    return max(max(abs(v) for v in clause) for clause in cnf), len(cnf)

def cnf_dump(cnf: List[List[int]]) -> str:
    ''' convert cnf expression to text data (can be saved to file) '''
    assert all(all(v != 0 for v in clause) for clause in cnf)
    variables, clauses = cnf_stats(cnf)
    result = f'p cnf {variables} {clauses}\n' \
        + ''.join(' '.join(map(str,clause)) + ' 0\n' for clause in cnf)
    return result

def cnf_load(cnf: str) -> List[List[int]]:
    ''' load cnf expression from text data (can be read from a file) '''
    lines = [line.split() for line in cnf.splitlines()]
    lines = [line for line in lines if len(line) > 0 and line[0] != 'c']
    assert len(lines) > 0 and len(lines[0]) == 4 and lines[0][:2] == ['p','cnf']
    variables = int(lines[0][2])
    clauses = int(lines[0][3])
    assert len(lines) == clauses+1
    result: List[List[int]] = []
    for line in lines[1:]:
        line = [int(v) for v in line]
        assert len(line) > 0 and line[-1] == 0
        assert all(1 <= abs(v) <= variables for v in line[:-1])
        result.append(line[:-1])
    return result

class TestDimacsCnf(unittest.TestCase):
    list1 = [[-1,2],[-1,3],[1,-2,-3]]
    text1 = 'p cnf 3 3\n-1 2 0\n-1 3 0\n1 -2 -3 0\n'
    def test_stats(self):
        self.assertEqual((3,3),cnf_stats(self.list1))
    def test_dump(self):
        self.assertEqual(self.text1,cnf_dump(self.list1))
    def test_load(self):
        self.assertEqual(self.list1,cnf_load(self.text1))

if __name__ == '__main__':
    unittest.main()
