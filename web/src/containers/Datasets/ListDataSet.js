import React from 'react';
import PropTypes from 'prop-types';
import { connect } from "react-redux"
import { compose } from "redux"
import { Link } from 'react-router-dom'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from '@material-ui/core/styles';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Tooltip from '@material-ui/core/Tooltip';
import IconButton from '@material-ui/core/IconButton';
import RefreshIcon from '@material-ui/icons/Refresh';

import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from "components/Console/ConsoleContents/ContentBar"

import ListDataSetTable from '../../components/Console/ConsoleContents/ListDataSetTable'

const styles = theme => ({
  block: {
    display: 'block',
  },
  addDS: {
    marginRight: theme.spacing(1),
  },
  contentWrapper: {
    margin: '40px 16px',
    //position: "relative",
    minHeight: 200,
  },
})


class ListDataSet extends React.Component {
  static propTypes = {
    classes: PropTypes.object.isRequired,
    handleHeaderTitleChange: PropTypes.func,
    DatasetList: PropTypes.array,
    requestListDS: PropTypes.func,
    resetLoadingBar: PropTypes.func,
  }

  state = {
    open: false,
    datasetSelected: "",
  }

  handleClose = () => {
    this.setState({
      open: false,
      datasetSelected: "",
    })
  }

  reloadSizeAndDS = () => {
    this.props.requestListDS()
  }

  componentDidMount() {
    this.props.handleHeaderTitleChange("Dataset > List Dataset")
    this.props.requestListDS()
  }

  componentDidUpdate(prevProps, prevState) {
    
  }

  componentWillUnmount() {
    this.props.resetLoadingBar()
  }

  render() {
    const {
      classes,
      DatasetList,
    } = this.props;

    return (
      <React.Fragment>
        <MainContent>
          <ContentBar>
            <Toolbar>
              <Grid container spacing={10} justify="space-between" alignItems="center">
                <Grid item>
                  <Typography variant="h5" gutterBottom>
                    List Dataset
                  </Typography>
                </Grid>
                <Grid item>
                  <Button
                    variant="contained"
                    color="primary"
                    className={classes.addDS}
                    component={Link}
                    to="/console/datasets/upload-datasets"
                  >
                    Add Dataset
                  </Button>
                  <Tooltip title="Reload">
                    <IconButton
                      onClick={this.reloadSizeAndDS}
                    >
                      <RefreshIcon className={classes.block} color="inherit" />
                    </IconButton>
                  </Tooltip>
                </Grid>
              </Grid>
            </Toolbar>
          </ContentBar>
          <div className={classes.contentWrapper}>
            <Typography color="textSecondary" align="center">
              {DatasetList.length === 0
                  ? "You do not have any dataset"
                  : "Datasets"
              }
            </Typography>
            <ListDataSetTable
              Datasets={DatasetList}
              handleClickHistory={this.handleClickHistory}
            />
          </div>
        </MainContent>
      </React.Fragment>
    )
  }
}

const mapStateToProps = state => ({
  DatasetList: state.DatasetsReducer.DatasetList,
})

const mapDispatchToProps = {
  handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
  requestListDS: actions.requestListDS,
  resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
  connect(mapStateToProps, mapDispatchToProps),
  withStyles(styles)
)(ListDataSet)
