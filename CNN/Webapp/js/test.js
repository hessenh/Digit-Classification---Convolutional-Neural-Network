$(document).ready(function(){


console.log("Halla bro")


	$("#send").click(function(e){
		$.ajax({
			url: "/CNN",
			method: "POST",
			beforeSend: function(xhr){
				var canvas = document.getElementById("canvas");
				var img    = canvas.toDataURL("image/png");

				console.log("nå sendes det")
			},

			data: "test"


		}).done(function(data){
			console.log(data)
		})
	})

})

