import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
import feas2
feas2.refine(sys.argv[1], int(sys.argv[2]), 128, int(sys.argv[3]))
