import { useState, useEffect, useRef} from 'react';
import Papa from 'papaparse';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { color } from 'chart.js/helpers';
import { Doughnut } from "react-chartjs-2";
import ChartDataLabels from "chartjs-plugin-datalabels";
import ReactSpeedometer from "react-d3-speedometer"
import './CommentsTable.css';
import './JumpingEmojis.css';


import './App.css';

ChartJS.register(ArcElement, Tooltip, Legend);
ChartJS.register(ChartDataLabels);




function App() {

  const [data, setData] = useState([]);
  const [columnData, setColumnData] = useState([]);
  const [subtopicData, setSubTopicData] = useState([]);
  const [chartData, setChartData] = useState({});
  const [selectedTopic, setSelectedTopic] = useState("");
  const [selectedComments, setSelectedComments] = useState([]);
  const [selectedSubTopics, setSelectedSubTopics] = useState([]);
  const [selectedSentiments, setSelectedSentiments] = useState([]);
  const [subTopicChartData, setSubTopicChartData] = useState({});
  const [showSubTopicChart, setShowSubTopicChart] = useState(false);
  const [SubTopicSelected, setSubTopicSelected] = useState("");
  const [hasTableData, sethasTableData] = useState(false);
  const [showSubTopicColumn, setShowSubTopicColumn] = useState(false);
  const [speedometerValue, setSpeedometerValue] = useState(0);
  const [positiveVal, setPositiveVal] = useState(0);
  const [negativeVal, setNegativeVal] = useState(0);
  const [neutralVal, setNeutralVal] = useState(0);
  const [PositiveAni, setPositiveAni] = useState(false);
  const [NegativeAni, setNegativeAni] = useState(false);
  const [NeutralAni, setNeutralAni] = useState(false);
  const [jumpCount, setJumpCount] = useState(0);
  const maxJumpCount = 5;

  const gradientColors = [
    "rgba(255, 99, 132, 1)",
    "rgba(54, 162, 235, 1)",
    // "rgba(255, 206, 86, 1)",
    "rgba(75, 192, 192, 1)",
    // "rgba(153, 102, 255, 1)",
    "rgba(255, 159, 64, 1)",
  ];

  const subTopicGradientColors = [
    "rgba(76, 63, 223, 0.8)",
    "rgba(230, 219, 66, 0.8)",
    "rgba(46, 200, 81, 0.8)",
    "rgba(255, 206, 86, 1)",
    "rgba(233, 117, 11, 0.8)",
  ];


  // parse CSV data & store it in the component state

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    Papa.parse(file, {
      header: true,
      complete: function (results) {
        setData(results.data);
      }
    });
  };



  useEffect(() => {
    if (data.length > 0) {
      const extractedColumnData = data.map((row) => row['Topics']);
      const extractedSubTopicData = data.map((row) => row['SubTopics']);
      const extractedSentimentsData = data.map((row) => row['Sentiment']);
      setColumnData(extractedColumnData);
      setSubTopicData(extractedSubTopicData);

      var p=0,n=0,neu=0;
      for(var i =0;i<extractedSentimentsData.length;i++)
      {
        if(extractedSentimentsData[i] === 'Positive')
          p++;
        else if(extractedSentimentsData[i] === 'Negative')
          n++;
        else
          neu++;    
      }

      if(p>n && p>neu)
      {
        setSpeedometerValue(50);
      }
      else if(n>p && n>neu)
      {
        setSpeedometerValue(10);
      }
      else if(neu>p && neu>n)
      {
        setSpeedometerValue(30);
      }
      else if(p==neu && neu!=n)
      {
        setSpeedometerValue(40);
      }
      else if(n==neu && neu!=p)
      {
        setSpeedometerValue(20);
      }
      else
      {
        setSpeedometerValue(30);
      }

    }
  }, [data]);

  useEffect(() => {
    if (columnData.length > 0) {
      const uniqueValues = [...new Set(columnData)];
      const counts = uniqueValues.map((value) =>
        columnData.filter((v) => v === value).length
      );
      // calculateRotationVal({counts})

      setChartData({
        labels: uniqueValues,
        datasets: [
          {
            data: counts,
            backgroundColor: function(context) {
              
              const colorIndex = context.dataIndex % gradientColors.length;
              let c = gradientColors[colorIndex];
          
                const mid = color(c).desaturate(0.4).darken(0.3).rgbString();
                const start = color(c).lighten(0.5).rotate(270).rgbString();
                const end = color(c).lighten(0.1).rgbString();
                return getGradient(context, start, mid, end);
            },
            datalabels:
            {
              rotation: function(ctx) {
                const valuesBefore = ctx.dataset.data.slice(0, ctx.dataIndex).reduce((a, b) => a + b, 0);
                const sum = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const rotation = ((valuesBefore + ctx.dataset.data[ctx.dataIndex] /2) /sum *360);
                return rotation < 180 ? rotation-90 : rotation+90;
              },
              
            }    
          }
]
      });
    }}, [columnData]);

    useEffect(() => {

    }, []);

    useEffect(() => {
      const interval = setInterval(startJumpAnimation, 2000);
      return () => clearInterval(interval);
    }, []);
  
    const startJumpAnimation = () => {
      setJumpCount(0);
      jumpAnimation();
    };
  
    const jumpAnimation = () => {
      setJumpCount((prevJumpCount) => prevJumpCount + 1);
  
      if (jumpCount < maxJumpCount) {
        setTimeout(jumpAnimation, 500);
      }
    };

    const CustomSegmentLabels = () => (
      <div>
    <div>
      <ReactSpeedometer
        width={300}
        needleHeightRatio={0.7}
        value={speedometerValue}
        customSegmentStops={[0,20,40,60]}
        segmentColors={['#FF4433', '#FAFA33', '#00FF00']}
        currentValueText="Sentiment Analysis"
        customSegmentLabels={[
          {
            text: 'Negative',
            position: 'INSIDE',
            color: '#555',
          },
          {
            text: 'Neutral',
            position: 'INSIDE',
            color: '#555',
          },
          {
            text: 'Positive',
            position: 'INSIDE',
            color: '#555',
          },
        ]}
        minValue={0}
        maxValue={60}
        ringWidth={47}
        needleTransitionDuration={0}
        needleTransition="easeElastic"
        needleColor={'#555'}
        textColor={'#d8dee9'}
        forceRender ={false}
      />
        </div>

        </div>)


  const handleSectionClick = (event, elements) => {
    if (elements && elements.length > 0) {
      console.log(elements[0].index);
      const index = elements[0].index;
      const topic = chartData.labels[index];
      setSelectedTopic(topic)

      const filteredResults = data.filter((row) => row['Topics'] === topic)
      setSelectedComments(filteredResults.map((item) => item.Comments));
      setSelectedSubTopics(filteredResults.map((item) => item.SubTopics));
      

      // setSelectedComments(comments);

      const subtopics = data.filter((row) => row['Topics'] === topic).map((row) => row['SubTopics']);



      if (subtopics.length > 0 && subtopics[0] != 'N/A') {
        const uniqueSubTopicsValues = [...new Set(subtopics)];
        const countsSubTopics = uniqueSubTopicsValues.map((value) =>
        subtopics.filter((v) => v === value).length
        );


      setSubTopicChartData({
        labels: uniqueSubTopicsValues,
        datasets: [
          {
            data: countsSubTopics,
            backgroundColor: function(context) {
              
              const colorIndex = context.dataIndex % subTopicGradientColors.length;
              let c = subTopicGradientColors[colorIndex];
          
                const mid = color(c).desaturate(0.4).darken(0.3).rgbString();
                const start = color(c).lighten(0.5).rotate(270).rgbString();
                const end = color(c).lighten(0.1).rgbString();
                return getGradient(context, start, mid, end);
            }, 
            datalabels:
            {
              rotation : function(ctx) {
                const valuesBefore = ctx.dataset.data.slice(0, ctx.dataIndex).reduce((a, b) => a + b, 0);
                const sum = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const rotation = ((valuesBefore + ctx.dataset.data[ctx.dataIndex] /2) /sum *360);
                return rotation < 180 ? rotation-90 : rotation+90;
              },
            }    
          }
        ]
      });
      setShowSubTopicChart(true);
      setShowSubTopicColumn(true);
    }
    sethasTableData(true);
  }};

  const handleSubTopicSectionClick = (event, elements) => {
    sethasTableData(false);
    setShowSubTopicChart(false);
    setShowSubTopicColumn(false);
  };

  const handleOnHoverEvent = (e, activeElements, chart) => {
    if (activeElements[0]) {
      let ctx = activeElements[0].element.$context;
      let label = chart.data.labels[ctx.dataIndex];
      setSubTopicSelected(label);
      // let value = chart.data.datasets[0].data[ctx.dataIndex];
    }
      
  };

  const handleTopicsOnHoverEvent = (e, activeElements, chart) => {
    if (activeElements[0]) {
      let ctx = activeElements[0].element.$context;
      let topic = chart.data.labels[ctx.dataIndex];
      const filteredResults = data.filter((row) => row['Topics'] === topic)
      setSelectedComments(filteredResults.map((item) => item.Comments));
      sethasTableData(true);

      setSelectedSentiments(filteredResults.map((item) => item.Sentiment));
      var p=0,n=0,neu=0;

      for(var i =0;i<selectedSentiments.length;i++)
      {
        if(selectedSentiments[i] === 'Positive')
          p++;
        else if(selectedSentiments[i] === 'Negative')
          n++;
        else
          neu++;    
      }

      if(p>n && p>neu)
      {
        setPositiveAni(true);
        setNegativeAni(false);
        setNeutralAni(false);
      }  
      else if(n>p && n>neu)
      {
        setNegativeAni(true);
        setNeutralAni(false);
        setPositiveAni(false);
      }
      else if(neu>p && neu>n)
      {
        setNeutralAni(true);
        setNegativeAni(false);
        setPositiveAni(false);
      }

      setPositiveVal(((p/(p+n+neu))*100).toFixed(2));
      setNegativeVal(((n/(p+n+neu))*100).toFixed(2));
      setNeutralVal(((neu/(p+n+neu))*100).toFixed(2));

      // if(p>n && p>neu)
      // {
      //   setSpeedometerValue(50);
      // }
      // else if(n>p && n>neu)
      // {
      //   setSpeedometerValue(10);
      // }
      // else if(neu>p && neu>n)
      // {
      //   setSpeedometerValue(30);
      // }
      // else if(p==neu && neu!=n)
      // {
      //   setSpeedometerValue(40);
      // }
      // else if(n==neu && neu!=p)
      // {
      //   setSpeedometerValue(20);
      // }
      // else
      // {
      //   setSpeedometerValue(30);
      // }


      // setSpeedometerValue(selectedComments.length);
      // let value = chart.data.datasets[0].data[ctx.dataIndex];
    }
      
  };


  const optionsSubChart = {
    plugins: {
      tooltip: {
        enabled: false,
      },
      legend: {
        display: false,

      },
      datalabels: {
        color: '#ffffff',
        display: true,
        formatter: function (value, context) {
          return context.chart.data.labels[context.dataIndex];
        },
        anchor : 'center',
        align : 'center',
        font: {
          weight: 'bold',
          size: 9.5
      }
      }
    },
    onClick: handleSubTopicSectionClick,
    onHover : handleOnHoverEvent
  };

  const options = {
    plugins: {
      tooltip: {
        enabled: false,
      },
      legend: {
        display: false,

      },
      datalabels: {
        color: '#ffffff',
        display: true,
        formatter: function (value, context) {
          return context.chart.data.labels[context.dataIndex];
        },
        anchor : 'center',
        align : 'center',
        font: {
          weight: 'bold',
          size: 9.5
      }
      },
    },
    onClick: handleSectionClick,
    onHover : handleTopicsOnHoverEvent,
  };

  function getGradient(context, c1, c2, c3)
  {
    const chartArea = context.chart.chartArea;
    if (!chartArea) {
      // This case happens on initial chart load
      return;
    }
    
    var gradient;
    // const ctx = context.chart.ctx;
    const {ctx, chartArea : {top,bottom, left, right} }= context.chart;
    gradient = ctx.createLinearGradient(left,top,right,bottom);
    gradient.addColorStop(0, c1);
    gradient.addColorStop(0.5, c2);
    gradient.addColorStop(1, c3);
    return gradient;
  }

  const addTextCenter = {
    id : 'textCenter',
    beforeDatasetsDraw(chart,args,pluginOptions){
      const {ctx,data} = chart;
      ctx.save();
      ctx.font = 'bolder 30px sans-serif';
      ctx.fillStyle = 'red';
      ctx.textAlign = 'center';
      ctx.fillText('text',chart.getDatasetMeta(0).data[0].x,chart.getDatasetMeta[0].data[0].y);
    }
  }


  return (
    <div className="App">

    <input type="file" accept=".csv" onChange={handleFileUpload} />

<div style={{ width: "550px", height: "500px", position: "absolute", top: "500px" }}>
{chartData.labels && chartData.datasets && !showSubTopicChart &&(
          <Doughnut
            data={chartData}
            options = {options}
            // plugins = {[addTextCenter]}
          />
        )}

      {showSubTopicChart && <Doughnut data={subTopicChartData} options = {optionsSubChart} />}
      </div >

  <div style={{ width: "550px", height: "500px", position: "absolute", top: "770px", left: "800px" }}>
  <h3>Selected Topic: {selectedTopic}</h3>


  {hasTableData && <table className="comments-table__table">
        <thead>
          <tr>
          {showSubTopicColumn &&<th>Subtopic</th>}
            <th>Comment</th>
          </tr>
        </thead>
        <tbody>
          {selectedComments.map((comments, index) => (
            <tr key={index}  className={selectedSubTopics[index] === SubTopicSelected ? 'comments-table__row--highlighted' : ''}>
              {showSubTopicColumn && <td>{selectedSubTopics[index]}</td>}
              <td>{comments}</td>
            </tr>
          ))}
        </tbody>
      </table>}
  </div>

<div>
  <h2 style={{position:"absolute" ,top:"60px", left:"60px"}}>Global Sentiment</h2>
  <div style={{ display: 'flex', justifyContent: 'flex-end', position:"absolute", left:"20px", top:"100px" }}>
      <CustomSegmentLabels />
    </div>
    </div>
<div style={{ position: "absolute", top: "500px", left: "800px" }}>
    <div className="jumping-emoji" style={{ position: 'absolute', top: "80px"}}>
    <div className="emoji1">
    {jumpCount > 0 && jumpCount <= maxJumpCount ? (
    <div className="jump-animation"></div>
  ) : (
    <div></div>
  )}
    </div>
    </div>
    <textarea style={{ width: "100px", height: "20px", position: "absolute", top: "200px"}} value={positiveVal} readOnly /> 

    <div className="jumping-emoji" style={{ position: 'absolute', top: '70px', left: '145px'}}> 
    <div className="emoji2">
     {jumpCount > 0 && jumpCount <= maxJumpCount && NeutralAni && (
      <div className="jump-animation"></div>
    )}
    </div> 
    </div>
    <textarea style={{ width: "100px", height: "20px", position: "absolute", top: "200px", left: "145px" }} value={neutralVal} readOnly /> 

    <div className="jumping-emoji" style={{ position: 'absolute', top: '70px', left: '300px'}}> 
    <div className= "emoji3"> 
    {jumpCount > 0 && jumpCount <= maxJumpCount && NegativeAni && (
      <div className="jump-animation"></div>
    )}
      </div>
      </div> 
      <textarea style={{ width: "100px", height: "20px", position: "absolute", top: "200px", left: "300px" }} value={negativeVal} readOnly /> 
  
</div>




    <br /><br />

</div>

  );
}

export default App;









