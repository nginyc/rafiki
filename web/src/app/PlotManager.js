/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

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
        // eslint-disable-next-line
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