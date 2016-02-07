class AddOnstageToCandidate < ActiveRecord::Migration
  def change
    add_column :candidates, :Onstage, :boolean, default: false
  end
end
