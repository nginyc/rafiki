import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

import { GeneralOptions, UtilityOptions } from "../Options"

const styles = {
  card: {
    maxWidth: "100%",
  },
};

function DocsCard(props) {
  const { classes } = props;
  return (
    <Card className={classes.card}>
      <CardContent>
        <Typography gutterBottom variant="h3" component="h1">
          Append
        </Typography>
        <Typography component="p">
          Append value at the end of the value for a key. The operation is not supported for data type "String".
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'APPEND -k <key> -x <value> [-b <branch> | -u <refer_version>]'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._x}
        {GeneralOptions._b}
        {GeneralOptions._u}
        <Typography component="p">
          Utility Options:
        </Typography>
        {UtilityOptions._none}
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
// Before APPEND
ustore> get -k File1 -b master
[SUCCESS: GET] Value<Blob>: "KEY,AGE,GENDER,GPA,SCHOOL
a01,20,Male,4.9,NUS
a02,22,Male,2.9,NTU
a03,28,Female,4.2,NUS
a04,39,Female,3.9,SMU
a05,16,Male,5.0,NUS"

// APPEND some value
ustore> append -k File1 -x " ...more...more...more" -b master
[SUCCESS: APPEND] Type: Blob, Version: TOAYEEWWILIDHXKF3CWURTGJIHQ2PTPR

// After APPEND
ustore> get -k File1 -b master
[SUCCESS: GET] Value<Blob>: "KEY,AGE,GENDER,GPA,SCHOOL
a01,20,Male,4.9,NUS
a02,22,Male,2.9,NTU
a03,28,Female,4.2,NUS
a04,39,Female,3.9,SMU
a05,16,Male,5.0,NUS ...more...more...more"
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
