$(document).ready(function(){


console.log("Halla bro")
	

	$("#send").click(function(e){
		$.ajax({
			url: "/CNN",
			method: "POST",
			data: {'number': document.getElementById("canvas").toDataURL("image/png")}


		}).done(function(data){
			console.log(data[0][1])
			document.getElementById("0").innerHTML = data[0][1]+" : " + data[0][0];
			document.getElementById("1").innerHTML = data[1][1]+" : " + data[1][0];
			document.getElementById("2").innerHTML = data[2][1]+" : " + data[2][0];
			document.getElementById("3").innerHTML = data[3][1]+" : " + data[3][0];
			document.getElementById("4").innerHTML = data[4][1]+" : " + data[4][0];
			document.getElementById("5").innerHTML = data[5][1]+" : " + data[5][0];
			document.getElementById("6").innerHTML = data[6][1]+" : " + data[6][0];
			document.getElementById("7").innerHTML = data[7][1]+" : " + data[7][0];
			document.getElementById("8").innerHTML = data[8][1]+" : " + data[8][0];
			document.getElementById("9").innerHTML = data[9][1]+" : " + data[9][0];


		})
	})

})

