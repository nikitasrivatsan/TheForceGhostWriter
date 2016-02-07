class CreateDebates < ActiveRecord::Migration
  def change
    create_table :debates do |t|
      t.string :p1
      t.string :p2
      t.integer :sL
      t.timestamps
    end
  end
end
