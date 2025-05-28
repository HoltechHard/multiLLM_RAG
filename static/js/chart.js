$(document).ready(function(){
    console.log("Initializing highcharts + jquery");

    if(typeof Highcharts == 'undefined'){
        console.error('highcharts library not loaded');
        $('#container').html('<p style="color:red;">highcharts library not loaded</p>');
        return;
    }

    try{
        Highcharts.chart('container', {
            chart: { type: 'bar' },
            title: { text: 'Historic World Population by Region' },
            xAxis: {
                categories: ['Africa', 'America', 'Asia', 'Europe'],
                title: { text: null },
                gridLineWidth: 1,
                lineWidth: 0
            },
            yAxis: {
                min: 0,
                title: { text: 'Population (millions)', align: 'high' },
                labels: { overflow: 'justify' },
                gridLineWidth: 0
            },
            tooltip: { valueSuffix: ' millions' },
            plotOptions: {
                bar: {
                    borderRadius: '50%',
                    dataLabels: { enabled: true },
                    groupPadding: 0.1
                }
            },
            legend: {
                layout: 'vertical',
                align: 'right',
                verticalAlign: 'top',
                x: -40,
                y: 80,
                floating: true,
                borderWidth: 1,
                backgroundColor: '#FFFFFF',
                shadow: true
            },
            credits: { enabled: false },
            series: [
                { name: 'Year 1990', data: [632, 727, 3202, 721] },
                { name: 'Year 2000', data: [814, 841, 3714, 726] },
                { name: 'Year 2021', data: [1393, 1031, 4695, 745] }
            ]
        });
        console.log('Chart rendered!');
    }catch(e){
        console.error('Chart rendering error: ', e);
        $('#container').html('<p style="color:red;">Chart rendering error: ' + e.message +'</p>');
    }
});
