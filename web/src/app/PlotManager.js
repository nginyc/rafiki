import echarts from 'echarts';
import _ from 'lodash';

export default class PlotManager {
  updatePlot(elemId, series, plotOption) {
    const dom = document.getElementById(elemId);
    if (!dom) {
      console.error(`Element of ID "${elemId}" doesn't exist on DOM!`);
      return;
    }

    const chartOption = this._getChartOption(series, plotOption);

    // @ts-ignore
    const chart = echarts.init(dom);  
    chart.setOption(chartOption);
  }

  _getChartOption(series, plotOption) {
    const xAxisName = _.get(plotOption, 'xAxis.name');
    const xAxisType = _.get(plotOption, 'xAxis.type', 'time');
    
    return {
      ...(plotOption.title ? {
        title: {
          text: plotOption.title
        }
      } : {}),
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type: (xAxisType == 'time') ? 'time': 'value',
        name: xAxisName,
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        splitLine: {
          show: false
        }
      },
      series: series.map(x => {
        return {
          name: x.name,
          type: 'line',
          data: x.data
        }
      })
    };
  }
}