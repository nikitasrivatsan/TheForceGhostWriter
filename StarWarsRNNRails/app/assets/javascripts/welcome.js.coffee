# Place all the behaviors and hooks related to the matching controller here.
# All this logic will automatically be available in application.js.
# You can use CoffeeScript in this file: http://coffeescript.org/


#JavaScript using JQuery & coffee - works in Chrome
$('#sL').on 'input', ->
  $('#target').text($('#sL').val())