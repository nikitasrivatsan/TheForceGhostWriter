# Python function
# input (president1, president2, textLength)
# ...
# ...
# output (speechtext)

# import pythontemplate
# call by: pythontemplate.textGenerator(p1,p2,tL)

def textGenerator(p1,p2,tL):
	s = ""
	for i in range(0,tL):
		s += p1 + " " + p2 + " "
	return s

# print (textGenerator("truman","reagan",50))

