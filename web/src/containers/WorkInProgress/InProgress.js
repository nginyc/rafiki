import React from 'react'

import * as ConsoleActions from "../ConsoleAppFrame/actions"

import { Paper } from "@material-ui/core"
import { connect } from "react-redux"

class InProgress extends React.Component {
    componentDidMount() {
        this.props.handleHeaderTitleChange("Working in progress")
    }

    render() {
        return (
            <main style={{
                flex: 1,
                padding: '48px 36px 0',
                background: '#eaeff1', // light grey
            }}>
                <Paper style={{
                    maxWidth: 1280,
                    margin: 'auto',
                    overflow: 'hidden',
                    marginBottom: 20,
                    position: "relative",
                    paddingBottom: 80,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                }}>
                        Working in Progress.
                </Paper>
            </main>
        )
    }
}

const mapStateToProps = state => ({
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
}

export default connect(mapStateToProps, mapDispatchToProps)(InProgress)