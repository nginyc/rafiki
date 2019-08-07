import React from 'react'

import { Link } from 'react-router-dom'

import { withStyles } from '@material-ui/core/styles'

// Material UI
import { Toolbar, Typography, Grid } from '@material-ui/core'
import RefreshIcon from '@material-ui/icons/Refresh'

// Import Layout
import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from 'components/Console/ConsoleContents/ContentBar'

import CreateTrainJobForm from "components/Console/ConsoleForms/CreateTrainJobForm";

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

class CreateTrainJob extends React.Component {
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
                                        Create Train Jobs
                                    </Typography>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentWrapper}>
                        <CreateTrainJobForm />
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}

export default withStyles(styles)(CreateTrainJob)