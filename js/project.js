$(document).ready(function(){
	$('.navbar-brand').hover(function() {
    	$('#lezhi').animate({
			width:"toggle",
			opacity:"toggle"
		});
    	// $('#lezhi', $(this)).toggle(500, 'linear').display(500, 'linear');
    });
});