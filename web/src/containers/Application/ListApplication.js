import React from 'react';
import PropTypes from 'prop-types';
import { connect } from "react-redux"
import { compose } from "redux"
import { Link } from 'react-router-dom'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"

import { withStyles } from '@material-ui/core/styles';
import { Button, Table, Toolbar, Typography, Grid, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core'
import moment from "moment"

import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from "components/Console/ConsoleContents/ContentBar"


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


class ListApplication extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        resetLoadingBar: PropTypes.func,
    }

    componentDidMount() {
        this.props.handleHeaderTitleChange("Application > List Application")
        this.props.getApplicationList()
    }

    componentDidUpdate(prevProps, prevState) {

    }

    componentWillUnmount() {
        this.props.resetLoadingBar()
    }

    render() {
        const {
            classes,
            ApplicationList
        } = this.props;

        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        List Application
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
                                        Open the page
                                    </Button>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentWrapper}>
                        <Typography color="textSecondary" align="center">
                            {ApplicationList.length === 0
                                ? "You do not have any trials for this job"
                                : "Jobs"
                            }
                        </Typography>
                        <Table>
                            <TableHead>
                                <TableRow>{
                                    ["ID", "App", "App Version", "Predictor Host",  "Status", "Started", "Stopped", "Duration", "Train Job ID"].map((label) => (<TableCell>{label}</TableCell>))
                                }
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {ApplicationList.map((x) => {
                                    return (
                                        <TableRow key={x.id} hover>
                                            <TableCell>{x.id}</TableCell>
                                            <TableCell>{x.app}</TableCell>
                                            <TableCell>{x.app_version}</TableCell>
                                            <TableCell>{x.predictor_host}</TableCell>
                                            <TableCell>{x.status}</TableCell>
                                            <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                                            <TableCell>{x.datetime_stopped ? moment(x.datetime_stopped).fromNow() : '-'}</TableCell>
                                            <TableCell>{
                                                x.datetime_stopped ?
                                                    moment.duration(x.datetime_stopped - x.datetime_started).humanize()
                                                    : '-'
                                            }</TableCell>
                                            <TableCell>{x.train_job_id}</TableCell>
                                        </TableRow>
                                    )
                                }
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}

const mapStateToProps = state => ({
    ApplicationList: state.ApplicationsReducer.ApplicationList
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
    getApplicationList: actions.fetchGetInferencejob
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(ListApplication)
