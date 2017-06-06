$(document).ready(function() {
            // $("#menu").load("menu.html");
            $("#show_border_check").on('change', function() {
            	if (this.checked) {
            		$("#border_in_between").hide();
            		$("#noborder_in_between").show();
            	}
            	else {
            		$("#border_in_between").show();
            		$("#noborder_in_between").hide();
            	}
            });
        });
