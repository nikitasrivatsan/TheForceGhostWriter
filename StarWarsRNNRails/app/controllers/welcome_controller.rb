class WelcomeController < ApplicationController
	def index
		if @p1 && @p2 && @tL # if all parameters are available
			@debate = `python lib/assets/pythontemplate/pythontemplate.py #{@p1} #{@p2} #{@tL}` 
		end

		@debate = Debate.find(1)
		
		@p1="truman"
		@p2="roosevelt"
		@tL=1
		#@speech = exec("python pythontemplate.py {p1,p2,sL}")
		@speech = `python lib/assets/pythontemplate/pythontemplate.py #{@p1} #{@p2} #{@tL}` 
	end
end
