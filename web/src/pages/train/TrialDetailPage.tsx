import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';

import { Paper, List, ListItem, Typography, Divider,
  CircularProgress, ListItemText } from '@material-ui/core';
import * as echarts from 'echarts';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';
import { TrialLogs, TrialPlot } from '../../../client/RafikiClient';

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
    this.updateCharts();
  }

  updateCharts() {
    const { logs } = this.state;

    if (!logs) return;
    
    this.charts = [];
    const chartOptions = getPlotChartOptions(logs);

    for (const i in chartOptions) {
      const chartOption = chartOptions[i];
      const dom = document.getElementById(`plot-${i}`);

      if (!dom) continue;

      // @ts-ignore
      const chart = echarts.init(dom);  
      chart.setOption(chartOption);
    }
  }

  renderLogs() {
    const { logs } = this.state;
    const { classes } = this.props;

    return (
      <div>
        {
          // Show plots section if there are plots
          Object.values(logs.plots).length > 0 &&
          <React.Fragment>
            <Typography gutterBottom variant="h3">Plots</Typography>
            {Object.values(logs.plots).map((x, i) => {
              return <Paper key={x.title} id={`plot-${i}`} className={classes.plotPaper}></Paper>;
            })}
          </React.Fragment>
        }
        {
          Object.values(logs.plots).length > 0 && Object.values(logs.messages).length > 0 &&
          <Divider className={classes.divider} />
        }
        {
          // Show messages section if there are messages
          Object.values(logs.messages).length > 0 &&
          <React.Fragment>
            <Typography gutterBottom variant="h3">Messages</Typography>
            <Paper>
              <List>
                {Object.values(logs.messages).map((x, i) => {
                  return (
                    <ListItem key={x.time + x.message}>
                      <ListItemText primary={x.message} secondary={x.time.toTimeString()} />
                    </ListItem>
                  );
                })}
                
              </List>
            </Paper>
          </React.Fragment>
        }
      </div>
    )
  }

  render() {
    const { classes, appUtils, trialId } = this.props;
    const { logs } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">
          Trial
          <span className={classes.headerSub}>{`(ID: V${trialId})`}</span>
        </Typography>
        <div className={classes.logsPaper}>
          {
            logs &&
            this.renderLogs()
          }
          {
            !logs &&
            <CircularProgress />
          }
        </div>          
      </React.Fragment>
    );
  }
}

function getPlotChartOptions(logs: TrialLogs): echarts.EChartOption[] {
  const chartOptions: echarts.EChartOption[] = [];
  console.log(logs);

  for (const plot of logs.plots) {
    const points = [];

    for (const metric of logs.metrics) {
      // Check if x axis value exists
      const xAxis = plot.x_axis || 'time';
      if (!(xAxis in metric)) {
        continue;
      }

      // Initialize point
      const point = [metric[xAxis]];

      // For each of plot's y axis metrics, add it to point data
      for (const plotMetric of plot.metrics) {
        point.push(plotMetric in metric ? metric[plotMetric] : null);
      }

      points.push(point);
    }

    const series = [];
    series.push({
      name,
      type: 'line',
      data: points
    });

    const chartOption: echarts.EChartOption = {
      title: {
        text: plot.title
      },
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type:  plot.x_axis ? 'value' : 'time',
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
    };

    chartOptions.push(chartOption);
  }

  return chartOptions;
}

const styles: StyleRulesCallback = (theme) => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing.unit * 2
  },
  logsPaper: {
    overflowX: 'auto'
  },
  plotPaper: {
    width: '100%',
    maxWidth: 800,
    height: 500,
    padding: theme.spacing.unit,
    paddingTop: theme.spacing.unit * 2,
    margin: theme.spacing.unit * 4
  },
  divider: {
    margin: theme.spacing.unit * 4
  }
});

export default withStyles(styles)(TrialDetailPage);