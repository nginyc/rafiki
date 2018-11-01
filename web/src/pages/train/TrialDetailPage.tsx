import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';

import { Paper, CircularProgress } from '@material-ui/core';
import * as echarts from 'echarts';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';
import { TrialLogs } from '../../../client/RafikiClient';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
  trialId: string;
}

class TrialDetailPage extends React.Component<Props> {
  charts: echarts.ECharts[] = [];
  state: {
    logs: TrialLogs | null
  } = {
    logs: null
  }

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError }, trialId } = this.props;
    try {
      const logs = await rafikiClient.getTrialLogs(trialId);
      this.setState({ logs });
    } catch (error) {
      showError(error, 'Failed to retrieve logs for trial');
    }
  }

  componentDidUpdate() {
    const { logs } = this.state;

    if (!logs) return;

    this.charts = [];
    const plots = Object.values(logs.plots);
    for (const i in plots) {
      const plot = plots[i];
      const dom = document.getElementById(`plot-${i}`);

      if (!dom) continue;

      const series = [];
      for (const name of plot.metrics) {
        series.push({
          name,
          type: 'line',
          data: logs.metrics[name].values.map(([date, value]) => {
            return {
              name: `(${date.toTimeString()}, ${value})`,
              value: [date, value]
            };
          })
        });
      }
      console.log(series);

      // @ts-ignore
      const chart = echarts.init(dom);  
      chart.setOption({
        title: {
          text: plot.title
        },
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'time',
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
        series
      });
    }
    
  }

  renderLogs() {
    const { logs } = this.state;
    const { classes } = this.props;

    return (
      <div>
        {Object.values(logs.plots).map((x, i) => {
          return <Paper key={x.title} id={`plot-${i}`} className={classes.plotPaper}></Paper>;
        })}
      </div>
    )
  }

  render() {
    const { classes, appUtils } = this.props;
    const { logs } = this.state;

    return (
      <main className={classes.main}>
          <Paper className={classes.logsPaper}>
            {
              logs &&
              this.renderLogs()
            }
            {
              !logs &&
              <CircularProgress />
            }
          </Paper>
          
      </main>
    );
  }
}

const styles: StyleRulesCallback = (theme) => ({
  main: {
  },
  logsPaper: {
    overflowX: 'auto'
  },
  plotPaper: {
    width: 500,
    height: 500
  }
});

export default withStyles(styles)(TrialDetailPage);