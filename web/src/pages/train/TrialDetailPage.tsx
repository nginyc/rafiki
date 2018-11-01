import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';

import { Paper, List, ListItem, Typography, Divider,
  CircularProgress, ListItemText } from '@material-ui/core';
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
    this.updateCharts();
  }

  updateCharts() {
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
                  const [date, text] = x;
                  return (
                    <ListItem>
                      <ListItemText primary={text} secondary={date.toTimeString()} />
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

const styles: StyleRulesCallback = (theme) => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing.unit * 2
  },
  logsPaper: {
    overflowX: 'auto'
  },
  plotPaper: {
    width: 500,
    height: 500
  },
  divider: {
    margin: 20
  }
});

export default withStyles(styles)(TrialDetailPage);