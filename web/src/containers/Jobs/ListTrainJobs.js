import React from 'react'
import PropTypes from 'prop-types'
import { Link } from 'react-router-dom'

import { connect } from 'react-redux'
import { compose } from "redux"

import { withStyles } from '@material-ui/core/styles'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as jobsActions from "./actions"

// Material UI
import { Table, Toolbar, Typography, Grid, Button, Tooltip, IconButton, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core'
import RefreshIcon from '@material-ui/icons/Refresh'

// Import Layout
import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from 'components/Console/ConsoleContents/ContentBar'


/* ListJobs are able to view trials and Trial details*/

const styles = theme => ({
    block: {
        display: 'block',
    },
    add: {
        marginRight: theme.spacing(1),
    },
    contentWrapper: {
        margin: '40px 16px',
        //position: "relative",
        minHeight: 200,
    },
})

class ListJobs extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        JobsList: PropTypes.array,
        requestJobsList: PropTypes.func,
        resetLoadingBar: PropTypes.func
    }

    state = {
        jobsSelected: "",
    }

    reloadJobs = () => {
        // TODO
    }

    componentDidMount() {
        this.props.handleHeaderTitleChange("Training Jobs > Jobs List")
        this.props.requestJobsList()
    }

    componentDidUpdate(prevProps, prevState) {

    }

    componentWillUnmount() {
        this.props.resetLoadingBar()
    }

    render() {
        const {
            classes
        } = this.props

        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={16} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        Training Jobs
                  </Typography>
                                </Grid>
                                <Grid item>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        className={classes.add}
                                        component={Link}
                                        to="/console/jobs/create-train-job"
                                    >
                                        Create New jobs
                  </Button>
                                    <Tooltip title="Reload">
                                        <IconButton
                                            onClick={console.log}
                                        >
                                            <RefreshIcon className={classes.block} color="inherit" />
                                        </IconButton>
                                    </Tooltip>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentWrapper}>
                        {/* <Typography color="textSecondary" align="center">
                            {DatasetList.length === 0
                                ? "You do not have any dataset"
                                : "Datasets"
                            }
                        </Typography> */}
                        <Table>
                            <TableHead>
                                <TableCell> App </TableCell>
                                <TableCell> App Version</TableCell>
                                <TableCell> Task </TableCell>
                                <TableCell> Budget </TableCell>
                                <TableCell> Started</TableCell>
                                <TableCell> Stopped </TableCell>
                                <TableCell> Status </TableCell>
                            </TableHead>
                            <TableBody></TableBody>
                        </Table>
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}

const mapStateToProps = state => ({
    JobsList: state.JobsReducer.jobsList
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    requestJobsList: jobsActions.requestJobsList,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
  }

export default compose(
    connect(mapStateToProps,mapDispatchToProps),
    withStyles(styles)
)(ListJobs)