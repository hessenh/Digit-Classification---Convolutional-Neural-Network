$(document).ready(function(){


console.log("Halla bro")
	

	$("#send").click(function(e){
		$.ajax({
			url: "/CNN",
			method: "POST",
			//beforeSend: function(xhr){
				//var canvas = document.getElementById("canvas");
				//var img    = document.getElementById("canvas").toDataURL("image/png");

			//	console.log("nå sendes det")
			//},

			data: {'number': document.getElementById("canvas").toDataURL("image/png")}


		}).done(function(data){
			console.log("S")

		})
	})

})

