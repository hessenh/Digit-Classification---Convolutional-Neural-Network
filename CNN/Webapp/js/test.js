$(document).ready(function(){


console.log("Halla bro")
	

	$("#send").click(function(e){
		$.ajax({
			url: "/CNN",
			method: "POST",
			data: {'number': document.getElementById("canvas").toDataURL("image/png")}


		}).done(function(data){
			document.getElementById("answer").innerHTML = "CNN:"+data;

		})
	})

})

