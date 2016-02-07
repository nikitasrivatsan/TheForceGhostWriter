# Python function
# input (president1, president2, textLength)
# ...
# ...
# output (speechtext)

# import pythontemplate
# call by: pythontemplate.textGenerator(p1,p2,tL)

import sys

def textGenerator(p1,p2,tL):
	s = ""
	for i in range(0,int(tL)):
		s += p1 + " " + p2 + " "
	return s

print (textGenerator(sys.argv[1],sys.argv[2],sys.argv[3]))
