class WelcomeController < ApplicationController
	def index
		# @result = exec("python script.py params")
		if @p1 && @p2 && @tL
			@debate = `python lib/assets/pythontemplate/pythontemplate.py #{@p1} #{@p2} #{@tL}` 
		end

		@p1="truman"
		@p2="roosevelt"
		@tL=1
		#@speech = exec("python pythontemplate.py {p1,p2,sL}")
		@debate = `python lib/assets/pythontemplate/pythontemplate.py #{@p1} #{@p2} #{@tL}` 
	end
end
