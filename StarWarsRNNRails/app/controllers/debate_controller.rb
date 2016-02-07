class DebateController < ApplicationController
	def show
	end

	def update
		@debate = Debate.find(1)
		@debate.update(debate_params)
	end

	private
	def debate_params
		params.require(:debate).permit(:p1,:p2,:sL)
	end
end
